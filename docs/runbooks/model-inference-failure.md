# Runbook: Model Inference Failure

## Alert Details

**Alert Name:** ModelInferenceFailure
**Severity:** Critical
**Component:** ML Pipeline
**Threshold:** Model inference error rate > 5% over 5 minutes

## Symptoms

- Model inference requests failing
- AlertManager firing ModelInferenceFailure alert
- GPU errors in logs
- Model loading failures
- Prediction timeouts
- CUDA out of memory errors
- Degraded prediction quality

## Diagnosis Steps

### 1. Check Model Inference Status

```bash
# Query inference error rate
curl -s 'http://prometheus:9090/api/v1/query?query=rate(sap_llm_model_inference_errors_total[5m])/rate(sap_llm_model_inference_requests_total[5m])' | jq .

# Check inference latency
curl -s 'http://prometheus:9090/api/v1/query?query=histogram_quantile(0.95,rate(sap_llm_model_inference_duration_bucket[5m]))' | jq .

# View model dashboard
open https://grafana.example.com/d/sap-llm-models

# Check error logs
kubectl logs -n sap-llm deployment/sap-llm --tail=100 | grep -i "inference\|model\|prediction"
```

### 2. Check GPU Status

```bash
# Check GPU availability
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

# Check GPU memory usage
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=memory.used,memory.total,memory.free --format=csv

# Check GPU utilization
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi dmon -c 5

# Check for GPU errors
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=gpu_name,ecc.errors.corrected.aggregate.total,ecc.errors.uncorrected.aggregate.total --format=csv

# Check CUDA availability
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "import torch; print(f'CUDA available: {torch.cuda.is_available()}, Devices: {torch.cuda.device_count()}')"
```

### 3. Check Model Loading Status

```bash
# Check if models are loaded
kubectl logs -n sap-llm deployment/sap-llm | grep "model loaded\|model loading\|model failed"

# Check model files exist
kubectl exec -it deployment/sap-llm -n sap-llm -- ls -lh /models/

# Verify model file integrity
kubectl exec -it deployment/sap-llm -n sap-llm -- sha256sum /models/*.bin

# Check model configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 10 "model"

# Test model endpoint
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -X POST http://localhost:8080/v1/predict -d '{"text":"test"}'
```

### 4. Check Resource Availability

```bash
# Check pod memory usage
kubectl top pods -n sap-llm

# Check GPU memory pressure
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=utilization.memory --format=csv,noheader,nounits

# Check disk space for model cache
kubectl exec -it deployment/sap-llm -n sap-llm -- df -h /models /tmp

# Check for resource limits
kubectl describe pod -n sap-llm -l app=sap-llm | grep -A 5 "Limits"
```

### 5. Check Model Dependencies

```bash
# Check Python/PyTorch versions
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 --version
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "import torch; print(f'PyTorch: {torch.__version__}')"

# Check CUDA driver version
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=driver_version --format=csv,noheader

# Check model library versions
kubectl exec -it deployment/sap-llm -n sap-llm -- pip list | grep -E "transformers|torch|cuda"

# Check for missing dependencies
kubectl logs -n sap-llm deployment/sap-llm | grep -i "import error\|module not found"
```

## Common Root Causes

### 1. GPU Memory Issues (40% of cases)

**Symptoms:** CUDA OOM errors, GPU memory exhausted, model loading failures

**Resolution:**
```bash
# Check GPU memory usage
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

# Clear GPU cache
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "
import torch
torch.cuda.empty_cache()
print('GPU cache cleared')
"

# Reduce batch size to free memory
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "inference_batch_size":"8",
  "max_batch_size":"16"
}}'

# Enable gradient checkpointing to save memory
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "use_gradient_checkpointing":"true",
  "model_precision":"fp16"
}}'

# Restart pods to clear GPU memory
kubectl rollout restart deployment/sap-llm -n sap-llm

# If persistent, scale down model instances
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "max_model_instances":"1"
}}'
```

### 2. Model Loading Failures (30% of cases)

**Symptoms:** Model not found, corrupted model files, version mismatches

**Resolution:**
```bash
# Check model files exist and are accessible
kubectl exec -it deployment/sap-llm -n sap-llm -- ls -lh /models/

# Verify model file checksums
kubectl exec -it deployment/sap-llm -n sap-llm -- sha256sum /models/*.bin

# Check model volume mount
kubectl describe pod -n sap-llm -l app=sap-llm | grep -A 10 "Mounts"

# Re-download model if corrupted
kubectl exec -it deployment/sap-llm -n sap-llm -- rm -rf /models/*
kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "
from transformers import AutoModel
model = AutoModel.from_pretrained('bert-base-uncased', cache_dir='/models')
print('Model downloaded')
"

# Check model configuration
kubectl get configmap sap-llm-config -n sap-llm -o yaml | grep -A 5 "model_name\|model_path"

# Update model path if incorrect
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "model_path":"/models/sap-llm-v2"
}}'

# Restart to reload model
kubectl rollout restart deployment/sap-llm -n sap-llm
```

### 3. CUDA/Driver Issues (20% of cases)

**Symptoms:** CUDA not available, driver errors, GPU not detected

**Resolution:**
```bash
# Check CUDA availability
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

# Check driver version compatibility
kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi --query-gpu=driver_version,cuda_version --format=csv

# Verify GPU device plugin
kubectl get daemonset -n kube-system | grep nvidia

# Check GPU resource allocation
kubectl describe node | grep -A 10 "nvidia.com/gpu"

# Restart GPU device plugin if needed
kubectl rollout restart daemonset nvidia-device-plugin-daemonset -n kube-system

# Restart pod to re-initialize CUDA
kubectl delete pod -n sap-llm -l app=sap-llm --field-selector=status.phase=Running --force

# If driver issue, drain and reboot node
kubectl drain <node-name> --ignore-daemonsets --delete-emptydir-data
# Reboot node
kubectl uncordon <node-name>
```

### 4. Model Quality/Performance Issues (10% of cases)

**Symptoms:** Low confidence scores, timeouts, degraded predictions

**Resolution:**
```bash
# Check inference latency
kubectl logs -n sap-llm deployment/sap-llm | grep "inference_duration_ms" | sort -t: -k2 -n | tail -20

# Check confidence scores
kubectl logs -n sap-llm deployment/sap-llm | grep "confidence_score" | awk -F: '{print $2}' | sort -n

# Increase timeout if needed
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "inference_timeout_seconds":"30"
}}'

# Switch to faster model variant
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "model_name":"sap-llm-fast",
  "model_precision":"fp16"
}}'

# Enable model optimization
kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{
  "use_tensorrt":"true",
  "optimize_for_inference":"true"
}}'

# Warm up model cache
kubectl exec -it deployment/sap-llm -n sap-llm -- curl -X POST http://localhost:8080/admin/warmup

# Restart to apply optimizations
kubectl rollout restart deployment/sap-llm -n sap-llm
```

## Resolution Steps

### Immediate Actions (0-5 minutes)

1. **Verify GPU Status:**
   ```bash
   # Check GPU health
   kubectl exec -it deployment/sap-llm -n sap-llm -- nvidia-smi

   # Check for OOM errors
   kubectl logs -n sap-llm deployment/sap-llm --tail=50 | grep -i "out of memory\|oom\|cuda"

   # Check if pods are running
   kubectl get pods -n sap-llm -l app=sap-llm
   ```

2. **Quick Recovery:**
   ```bash
   # Clear GPU cache
   kubectl exec -it deployment/sap-llm -n sap-llm -- python3 -c "import torch; torch.cuda.empty_cache()"

   # Restart problematic pods
   kubectl delete pod -n sap-llm $(kubectl get pods -n sap-llm -l app=sap-llm -o jsonpath='{.items[?(@.status.containerStatuses[0].restartCount>0)].metadata.name}')

   # Test inference endpoint
   kubectl exec -it deployment/sap-llm -n sap-llm -- curl -X POST http://localhost:8080/v1/predict -d '{"text":"test"}'
   ```

3. **Reduce Load:**
   ```bash
   # Reduce batch size temporarily
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"inference_batch_size":"4"}}'

   # Limit concurrent requests
   kubectl patch configmap sap-llm-config -n sap-llm --type merge -p '{"data":{"max_concurrent_inferences":"10"}}'

   # Restart to apply changes
   kubectl rollout restart deployment/sap-llm -n sap-llm
   ```

### Short-term Fixes (5-30 minutes)

1. **If GPU Memory Issue:**
   - Reduce batch sizes
   - Enable FP16 precision
   - Limit model instances
   - Clear unused models

2. **If Model Loading Issue:**
   - Verify model files
   - Re-download if corrupted
   - Check permissions
   - Validate configuration

3. **If CUDA Issue:**
   - Restart GPU plugin
   - Restart pods
   - Check driver version
   - Reboot node if needed

### Long-term Solutions (30+ minutes)

1. **Model Optimization:**
   - Implement model quantization
   - Use TensorRT optimization
   - Enable gradient checkpointing
   - Use distilled models

2. **Infrastructure:**
   - Upgrade GPU drivers
   - Add more GPU nodes
   - Implement GPU sharing
   - Use mixed precision training

3. **Monitoring:**
   - Add GPU metrics
   - Track inference quality
   - Monitor model performance
   - Set up alerting

## Escalation

### Level 1: On-Call Engineer (0-15 minutes)
- Follow this runbook
- Restart pods and clear caches
- Verify GPU availability
- Test inference endpoint

### Level 2: ML Engineer (15-30 minutes)
- If model loading issues
- If quality degradation
- If configuration needed
- If model optimization required

### Level 3: ML Architecture Team (30+ minutes)
- If model changes needed
- If GPU driver issues
- If infrastructure changes required
- If persistent failures

## Prevention

1. **Pre-deployment:**
   - Test model loading
   - Validate GPU compatibility
   - Load test inference
   - Verify model checksums

2. **Monitoring:**
   - GPU health checks
   - Inference error rates
   - Model performance metrics
   - Resource utilization

3. **Configuration:**
   - Appropriate batch sizes
   - Memory limits
   - Timeout settings
   - Model warm-up

4. **Testing:**
   - GPU stress testing
   - Model validation tests
   - Performance regression tests
   - Failure scenario testing

## Related Runbooks

- [High Memory](./high-memory.md)
- [GPU Utilization Low](./gpu-utilization-low.md)
- [High Latency](./high-latency.md)
- [High Error Rate](./high-error-rate.md)

## Revision History

| Date | Author | Changes |
|------|--------|---------|
| 2025-11-17 | SAP_LLM Team | Initial version |
