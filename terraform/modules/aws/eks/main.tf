# AWS EKS Module (Multi-region fallback)

data "aws_availability_zones" "available" {
  state = "available"
}

# VPC for EKS
module "vpc" {
  source  = "terraform-aws-modules/vpc/aws"
  version = "~> 5.0"

  name = "sap-llm-${var.environment}-vpc"
  cidr = "10.2.0.0/16"

  azs             = slice(data.aws_availability_zones.available.names, 0, 3)
  private_subnets = ["10.2.1.0/24", "10.2.2.0/24", "10.2.3.0/24"]
  public_subnets  = ["10.2.11.0/24", "10.2.12.0/24", "10.2.13.0/24"]

  enable_nat_gateway   = true
  single_nat_gateway   = false
  enable_dns_hostnames = true
  enable_dns_support   = true

  public_subnet_tags = {
    "kubernetes.io/role/elb" = 1
  }

  private_subnet_tags = {
    "kubernetes.io/role/internal-elb" = 1
  }

  tags = var.tags
}

# EKS Cluster
module "eks" {
  source  = "terraform-aws-modules/eks/aws"
  version = "~> 19.0"

  cluster_name    = "sap-llm-${var.environment}-eks"
  cluster_version = var.kubernetes_version

  vpc_id                   = module.vpc.vpc_id
  subnet_ids               = module.vpc.private_subnets
  control_plane_subnet_ids = module.vpc.public_subnets

  # Cluster access
  cluster_endpoint_public_access  = true
  cluster_endpoint_private_access = true

  # Encryption
  cluster_encryption_config = {
    provider_key_arn = aws_kms_key.eks.arn
    resources        = ["secrets"]
  }

  # Managed node groups
  eks_managed_node_groups = {
    # General purpose nodes
    general = {
      name           = "general"
      instance_types = ["m5.2xlarge"]

      min_size     = var.node_count
      max_size     = var.node_count * 5
      desired_size = var.node_count

      disk_size = 100

      labels = {
        role = "general"
      }

      tags = var.tags
    }

    # GPU nodes
    gpu = {
      name           = "gpu"
      instance_types = ["p3.2xlarge"]  # Tesla V100

      min_size     = 1
      max_size     = 10
      desired_size = 2

      disk_size = 200

      labels = {
        role        = "gpu"
        accelerator = "nvidia-tesla-v100"
      }

      taints = [{
        key    = "nvidia.com/gpu"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = var.tags
    }

    # Spot instances
    spot = {
      name           = "spot"
      instance_types = ["m5.xlarge", "m5a.xlarge", "m5n.xlarge"]
      capacity_type  = "SPOT"

      min_size     = 2
      max_size     = 20
      desired_size = 5

      labels = {
        role = "spot"
      }

      taints = [{
        key    = "spot"
        value  = "true"
        effect = "NO_SCHEDULE"
      }]

      tags = var.tags
    }
  }

  # Cluster addons
  cluster_addons = {
    coredns = {
      most_recent = true
    }
    kube-proxy = {
      most_recent = true
    }
    vpc-cni = {
      most_recent = true
    }
    aws-ebs-csi-driver = {
      most_recent = true
    }
  }

  tags = var.tags
}

# KMS key for EKS encryption
resource "aws_kms_key" "eks" {
  description             = "EKS cluster encryption key"
  deletion_window_in_days = 7
  enable_key_rotation     = true

  tags = var.tags
}

resource "aws_kms_alias" "eks" {
  name          = "alias/sap-llm-${var.environment}-eks"
  target_key_id = aws_kms_key.eks.key_id
}

# Variables
variable "environment" {
  type = string
}

variable "region" {
  type = string
}

variable "kubernetes_version" {
  type = string
}

variable "node_count" {
  type = number
}

variable "tags" {
  type = map(string)
}

# Outputs
output "cluster_name" {
  value = module.eks.cluster_name
}

output "cluster_endpoint" {
  value = module.eks.cluster_endpoint
}

output "cluster_certificate_authority_data" {
  value     = module.eks.cluster_certificate_authority_data
  sensitive = true
}
