# Azure Kubernetes Service (AKS) Module

resource "azurerm_resource_group" "aks" {
  name     = "sap-llm-${var.environment}-aks"
  location = var.location
  tags     = var.tags
}

resource "azurerm_kubernetes_cluster" "main" {
  name                = "sap-llm-${var.environment}-aks"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  dns_prefix          = "sap-llm-${var.environment}"
  kubernetes_version  = var.kubernetes_version

  # Default node pool (CPU workloads)
  default_node_pool {
    name                = "default"
    node_count          = var.node_count
    vm_size             = "Standard_D8s_v5"
    type                = "VirtualMachineScaleSets"
    availability_zones  = ["1", "2", "3"]
    enable_auto_scaling = true
    min_count           = var.node_count
    max_count           = var.node_count * 5
    os_disk_size_gb     = 128
    os_disk_type        = "Managed"

    upgrade_settings {
      max_surge = "33%"
    }

    tags = var.tags
  }

  # Identity
  identity {
    type = "SystemAssigned"
  }

  # Network profile
  network_profile {
    network_plugin     = "azure"
    network_policy     = "calico"
    load_balancer_sku  = "standard"
    service_cidr       = "10.0.0.0/16"
    dns_service_ip     = "10.0.0.10"
  }

  # Azure AD integration
  azure_active_directory_role_based_access_control {
    managed                = true
    azure_rbac_enabled     = true
    admin_group_object_ids = []
  }

  # Security
  role_based_access_control_enabled = true

  # Monitoring
  oms_agent {
    log_analytics_workspace_id = azurerm_log_analytics_workspace.aks.id
  }

  # Autoscaler profile
  auto_scaler_profile {
    balance_similar_node_groups      = true
    expander                         = "least-waste"
    max_graceful_termination_sec     = 600
    max_node_provisioning_time       = "15m"
    scale_down_delay_after_add       = "10m"
    scale_down_unneeded              = "10m"
    scale_down_utilization_threshold = 0.5
  }

  tags = var.tags
}

# GPU Node Pool
resource "azurerm_kubernetes_cluster_node_pool" "gpu" {
  name                  = "gpu"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_NC6s_v3"  # Tesla V100
  node_count            = var.gpu_node_count
  availability_zones    = ["1", "2", "3"]
  enable_auto_scaling   = true
  min_count             = 1
  max_count             = var.gpu_node_count * 3
  os_disk_size_gb       = 256
  os_disk_type          = "Managed"

  node_labels = {
    "workload"    = "gpu-intensive"
    "accelerator" = "nvidia-tesla-v100"
  }

  node_taints = [
    "nvidia.com/gpu=true:NoSchedule"
  ]

  tags = var.tags
}

# Spot Instance Node Pool (Cost Optimization)
resource "azurerm_kubernetes_cluster_node_pool" "spot" {
  name                  = "spot"
  kubernetes_cluster_id = azurerm_kubernetes_cluster.main.id
  vm_size               = "Standard_D4s_v5"
  priority              = "Spot"
  eviction_policy       = "Delete"
  spot_max_price        = -1  # Pay up to on-demand price

  enable_auto_scaling = true
  min_count           = 2
  max_count           = 20
  os_disk_size_gb     = 128

  node_labels = {
    "kubernetes.azure.com/scalesetpriority" = "spot"
    "workload" = "batch-processing"
  }

  node_taints = [
    "kubernetes.azure.com/scalesetpriority=spot:NoSchedule"
  ]

  tags = var.tags
}

# Log Analytics Workspace
resource "azurerm_log_analytics_workspace" "aks" {
  name                = "sap-llm-${var.environment}-logs"
  location            = azurerm_resource_group.aks.location
  resource_group_name = azurerm_resource_group.aks.name
  sku                 = "PerGB2018"
  retention_in_days   = 30

  tags = var.tags
}

# Container Insights Solution
resource "azurerm_log_analytics_solution" "container_insights" {
  solution_name         = "ContainerInsights"
  location              = azurerm_resource_group.aks.location
  resource_group_name   = azurerm_resource_group.aks.name
  workspace_resource_id = azurerm_log_analytics_workspace.aks.id
  workspace_name        = azurerm_log_analytics_workspace.aks.name

  plan {
    publisher = "Microsoft"
    product   = "OMSGallery/ContainerInsights"
  }

  tags = var.tags
}

# Variables
variable "environment" {
  type = string
}

variable "location" {
  type = string
}

variable "kubernetes_version" {
  type = string
}

variable "node_count" {
  type = number
}

variable "gpu_node_count" {
  type = number
}

variable "tags" {
  type = map(string)
}

# Outputs
output "cluster_name" {
  value = azurerm_kubernetes_cluster.main.name
}

output "cluster_id" {
  value = azurerm_kubernetes_cluster.main.id
}

output "kube_config" {
  value     = azurerm_kubernetes_cluster.main.kube_config_raw
  sensitive = true
}
