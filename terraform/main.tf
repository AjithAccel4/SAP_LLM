"""
ENHANCEMENT 3: Infrastructure as Code (Terraform Multi-Cloud)

Multi-cloud infrastructure provisioning:
- Azure Kubernetes Service (AKS)
- AWS EKS as fallback
- GCP GKE for multi-region
- Cosmos DB, Redis, Storage
- Network infrastructure
- Security groups and policies
"""

terraform {
  required_version = ">= 1.5.0"

  required_providers {
    azurerm = {
      source  = "hashicorp/azurerm"
      version = "~> 3.0"
    }
    aws = {
      source  = "hashicorp/aws"
      version = "~> 5.0"
    }
    google = {
      source  = "hashicorp/google"
      version = "~> 5.0"
    }
    kubernetes = {
      source  = "hashicorp/kubernetes"
      version = "~> 2.23"
    }
    helm = {
      source  = "hashicorp/helm"
      version = "~> 2.11"
    }
  }

  backend "azurerm" {
    resource_group_name  = "sap-llm-terraform"
    storage_account_name = "sapllmtfstate"
    container_name       = "tfstate"
    key                  = "production.terraform.tfstate"
  }
}

# Azure Provider
provider "azurerm" {
  features {
    key_vault {
      purge_soft_delete_on_destroy = false
    }
  }
}

# AWS Provider
provider "aws" {
  region = var.aws_region
}

# GCP Provider
provider "google" {
  project = var.gcp_project_id
  region  = var.gcp_region
}

# Variables
variable "environment" {
  description = "Environment name"
  type        = string
  default     = "production"
}

variable "location" {
  description = "Azure region"
  type        = string
  default     = "eastus"
}

variable "aws_region" {
  description = "AWS region"
  type        = string
  default     = "us-east-1"
}

variable "gcp_project_id" {
  description = "GCP project ID"
  type        = string
}

variable "gcp_region" {
  description = "GCP region"
  type        = string
  default     = "us-central1"
}

variable "kubernetes_version" {
  description = "Kubernetes version"
  type        = string
  default     = "1.28"
}

variable "node_count" {
  description = "Number of nodes in default pool"
  type        = number
  default     = 3
}

variable "gpu_node_count" {
  description = "Number of GPU nodes"
  type        = number
  default     = 2
}

# Locals
locals {
  common_tags = {
    Project     = "SAP_LLM"
    Environment = var.environment
    ManagedBy   = "Terraform"
    CostCenter  = "AI-ML"
  }
}

# Modules
module "azure_aks" {
  source = "./modules/azure/aks"

  environment         = var.environment
  location            = var.location
  kubernetes_version  = var.kubernetes_version
  node_count          = var.node_count
  gpu_node_count      = var.gpu_node_count
  tags                = local.common_tags
}

module "azure_cosmos" {
  source = "./modules/azure/cosmos"

  environment = var.environment
  location    = var.location
  tags        = local.common_tags
}

module "azure_storage" {
  source = "./modules/azure/storage"

  environment = var.environment
  location    = var.location
  tags        = local.common_tags
}

module "azure_networking" {
  source = "./modules/azure/networking"

  environment = var.environment
  location    = var.location
  tags        = local.common_tags
}

module "aws_eks" {
  source = "./modules/aws/eks"

  environment        = var.environment
  region             = var.aws_region
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  tags               = local.common_tags
}

module "gcp_gke" {
  source = "./modules/gcp/gke"

  project_id         = var.gcp_project_id
  environment        = var.environment
  region             = var.gcp_region
  kubernetes_version = var.kubernetes_version
  node_count         = var.node_count
  labels             = local.common_tags
}

# Outputs
output "azure_aks_cluster_name" {
  value = module.azure_aks.cluster_name
}

output "azure_cosmos_endpoint" {
  value     = module.azure_cosmos.endpoint
  sensitive = true
}

output "aws_eks_cluster_name" {
  value = module.aws_eks.cluster_name
}

output "gcp_gke_cluster_name" {
  value = module.gcp_gke.cluster_name
}
