# Runbook: Low GPU Utilization

## Alert Details

**Alert Name:** GPUUtilizationLow
**Severity:** Warning
**Component:** ML Pipeline
**Threshold:** GPU utilization < 30% over 15 minutes

## Symptoms

- GPU utilization consistently below 30%
- AlertManager firing GPUUtilizationLow alert
- Expensive GPU resources underutilized
- Poor cost efficiency
- Potential throughput issues
- Model inference not fully leveraging GPU

## Diagnosis Steps

### 1. Check Current GPU Utilization

```bash
# Check real-time GPU utilization
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi dmon -c 10

# Get detailed GPU stats
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=utilization.gpu,utilization.memory,memory.used,memory.total --format=csv

# Query Prometheus for GPU metrics
curl -s 'http://prometheus:9090/api/v1/query?query=avg(sap_llm_gpu_utilization_percent)' | jq .

# View GPU dashboard
open https://grafana.example.com/d/sap-llm-gpu

# Check utilization over time
curl -s 'http://prometheus:9090/api/v1/query_range?query=sap_llm_gpu_utilization_percent&start=-1h&step=60s' | jq .
```

### 2. Check Inference Load

```bash
# Check inference request rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_model_inference_requests_total[5m])' | jq .

# Check active inference tasks
kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep "inference\|prediction"

# Check queue depth
kubectl exec -it deployment/redis -n sap-llm -- redis-cli LLEN sap_llm:inference_queue

# Check concurrent inference requests
curl -s 'http://prometheus:9090/api/v1/query?query=sap_llm_inference_requests_in_progress' | jq .
```

### 3. Check Batch Processing Configuration

```bash
# Check current batch size
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep "batch_size"

# Check batch configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 5 "batch"

# Check batch wait times
kubectl logs -n sap-llm deployment/sap-llm | grep "batch_wait_ms"

# Check if batching is enabled
kubectl exec -it deployment/sap-llm -n sap-llm -- env | grep -i batch
```

### 4. Check CPU vs GPU Balance

```bash
# Check CPU utilization
kubectl top pods -n sap-llm

# Check if CPU is bottleneck
kubectl exec -it deployment/sap-llm -n sap-llm -- top -b -n 1 | head -20

# Check data loading times
kubectl logs -n sap-llm deployment/sap-llm | grep "data_load_duration_ms"

# Check preprocessing times
kubectl logs -n sap-llm deployment/sap-llm | grep "preprocessing_duration_ms"
```

### 5. Check Model Configuration

```bash
# Check model precision
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -i "precision\|quantization"

# Check if model is GPU-enabled
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "
import torch
print(f'Model device: {next(iter(model.parameters())).device}')
"

# Check model optimization settings
kubectl logs -n sap-llm deployment/sap-llm | grep -i "tensorrt\|optimization\|fp16"

# Check inference mode
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 3 "inference_mode"
```

## Common Root Causes

### 1. Small Batch Sizes (40% of cases)

**Symptoms:** Frequent small batches, low GPU saturation, high inference overhead

**Resolution:**
```bash
# Check current batch size
kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep "batch_size" | sort -t: -k2 -n

# Increase batch size
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "inference_batch_size":"64",
  "max_batch_size":"128"
}}'

# Increase batch wait time to accumulate more requests
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "batch_wait_ms":"200",
  "min_batch_size":"16"
}}'

# Enable dynamic batching
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "enable_dynamic_batching":"true",
  "dynamic_batch_timeout_ms":"100"
}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm

# Monitor utilization improvement
watch -n 5 'kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'
```

### 2. CPU Bottleneck (30% of cases)

**Symptoms:** High CPU usage, slow data loading, preprocessing delays

**Resolution:**
```bash
# Check CPU utilization
kubectl top pods -n sap-llm

# Increase CPU resources
kubectl patch deployment sap-llm -n sap-llm --type json -p='[
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/limits/cpu", "value": "8000m"},
  {"op": "replace", "path": "/spec/template/spec/containers/0/resources/requests/cpu", "value": "4000m"}
]'

# Increase data loader workers
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "num_workers":"8",
  "prefetch_factor":"4"
}}'

# Enable pinned memory for faster data transfer
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "pin_memory":"true",
  "non_blocking":"true"
}}'

# Optimize preprocessing
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "preprocess_on_gpu":"true",
  "use_fast_tokenizer":"true"
}}'

# Restart to apply changes
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. Low Traffic/Load (20% of cases)

**Symptoms:** Few inference requests, idle periods, queue empty

**Resolution:**
```bash
# Check request rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_model_inference_requests_total[15m])*60' | jq .

# Check if this is expected load
kubectl logs -n sap-llm deployment/sap-llm --since=1h | grep -c "inference_request"

# Scale down GPU pods if consistently low traffic
kubectl scale deployment sap-llm -n sap-llm --replicas=2

# Consider scheduling GPU pods during peak hours only
kubectl patch deployment sap-llm -n sap-llm -p '{"spec":{"template":{"metadata":{"annotations":{
  "cluster-autoscaler.kubernetes.io/safe-to-evict": "true"
}}}}}'

# Implement GPU pod autoscaling based on queue depth
kubectl autoscale deployment sap-llm -n sap-llm \
  --min=1 --max=5 \
  --cpu-percent=60

# Consider spot instances for cost optimization
kubectl patch deployment sap-llm -n sap-llm --type json -p='[
  {"op": "add", "path": "/spec/template/spec/nodeSelector", "value": {"node.kubernetes.io/instance-type": "g4dn.xlarge-spot"}}
]'

# Monitor cost vs performance
# Review GPU node utilization across all workloads
kubectl describe nodes -l node.kubernetes.io/instance-type=g4dn.xlarge
```

### 4. Model Not Optimized for GPU (10% of cases)

**Symptoms:** Low compute intensity, inefficient GPU operations, poor kernel utilization

**Resolution:**
```bash
# Check model precision
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -i precision

# Enable FP16 for better GPU utilization
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "model_precision":"fp16",
  "use_amp":"true"
}}'

# Enable TensorRT optimization
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "use_tensorrt":"true",
  "tensorrt_precision":"fp16"
}}'

# Use optimized CUDA kernels
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "use_flash_attention":"true",
  "use_cuda_graphs":"true"
}}'

# Enable kernel fusion
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "enable_kernel_fusion":"true",
  "optimize_for_inference":"true"
}}'

# Compile model with optimizations
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "
import torch
# Compile and optimize model
model = torch.compile(model, mode='max-autotune')
"

# Restart to apply optimizations
kubectl rollout restart deployment/sap-llm -n sap-llm

# Verify improved utilization
watch -n 2 'kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi'
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Verify Low Utilization:**
   ```bash
   # Confirm GPU utilization is actually low
   kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi dmon -c 5

   # Check if this is due to low traffic
   curl http://prometheus:9090/api/v1/query?query=rate(sap_llm_model_inference_requests_total[5m])

   # Check batch sizes
   kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep "batch_size"
   ```

2. **Quick Configuration Adjustments:**
   ```bash
   # Increase batch size
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"inference_batch_size":"64"}}'

   # Increase batch wait time
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"batch_wait_ms":"200"}}'

   # Restart pods
   kubectl rollout restart deployment/sap-llm -n sap-llm
   ```

3. **Monitor Changes:**
   ```bash
   # Watch GPU utilization
   watch -n 5 'kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=utilization.gpu --format=csv,noheader'

   # Monitor batch sizes
   kubectl logs -n sap-llm deployment/sap-llm -f | grep "batch_size"
   ```

### Short-term Fixes (5-30 minutes)

1. **If Small Batches:**
   - Increase batch sizes
   - Enable dynamic batching
   - Increase batch wait times
   - Implement request queuing

2. **If CPU Bottleneck:**
   - Increase CPU allocation
   - Add more workers
   - Optimize preprocessing
   - Move preprocessing to GPU

3. **If Low Traffic:**
   - Scale down GPU pods
   - Consider pod scheduling
   - Implement autoscaling
   - Evaluate cost optimization

### Long-term Solutions (30+ minutes)

1. **Model Optimization:**
   - Implement FP16/INT8 precision
   - Enable TensorRT
   - Use CUDA optimizations
   - Implement model compilation

2. **Architecture:**
   - Implement request batching service
   - Add load balancing
   - Optimize data pipeline
   - Consider model serving framework

3. **Cost Optimization:**
   - Right-size GPU instances
   - Use spot instances
   - Implement time-based scaling
   - Share GPU across workloads

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Adjust batch sizes
- Check traffic levels
- Monitor utilization changes

### Level 2: ML Engineer (15-30 minutes)
- If optimization needed
- If model configuration changes required
- If performance tuning needed
- If architecture review needed

### Level 3: ML Architecture Team (30+ minutes)
- If model redesign needed
- If infrastructure changes required
- If cost optimization strategy needed
- If new serving framework needed

## Prevention

1. **Monitoring:**
   - Track GPU utilization trends
   - Monitor batch size distributions
   - Alert on sustained low utilization
   - Track cost per inference

2. **Configuration:**
   - Set appropriate batch sizes
   - Configure dynamic batching
   - Optimize CPU resources
   - Use mixed precision

3. **Load Testing:**
   - Test with various batch sizes
   - Validate GPU utilization
   - Benchmark performance
   - Profile GPU kernels

4. **Cost Management:**
   - Regular utilization reviews
   - Right-size GPU instances
   - Implement autoscaling
   - Use spot instances when appropriate

## Related Runbooks

- [Model Inference Failure](./model-inference-failure.md)
- [Low Throughput](./low-throughput.md)
- [High Latency](./high-latency.md)
- [High Memory](./high-memory.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
