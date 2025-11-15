# Azure Cosmos DB Module (Gremlin API for PMG)

resource "azurerm_resource_group" "cosmos" {
  name     = "sap-llm-${var.environment}-cosmos"
  location = var.location
  tags     = var.tags
}

resource "azurerm_cosmosdb_account" "main" {
  name                = "sap-llm-${var.environment}-cosmos"
  location            = azurerm_resource_group.cosmos.location
  resource_group_name = azurerm_resource_group.cosmos.name
  offer_type          = "Standard"
  kind                = "GlobalDocumentDB"

  # Multi-region configuration
  geo_location {
    location          = var.location
    failover_priority = 0
    zone_redundant    = true
  }

  geo_location {
    location          = "westus"
    failover_priority = 1
    zone_redundant    = true
  }

  geo_location {
    location          = "westeurope"
    failover_priority = 2
    zone_redundant    = true
  }

  # Consistency
  consistency_policy {
    consistency_level       = "Session"
    max_interval_in_seconds = 5
    max_staleness_prefix    = 100
  }

  # Capabilities
  capabilities {
    name = "EnableGremlin"
  }

  capabilities {
    name = "EnableServerless"
  }

  # Backup
  backup {
    type                = "Continuous"
    interval_in_minutes = 240
    retention_in_hours  = 720  # 30 days
  }

  # Security
  is_virtual_network_filter_enabled = true

  virtual_network_rule {
    id = azurerm_subnet.cosmos.id
  }

  # Monitoring
  analytical_storage_enabled = true

  tags = var.tags
}

# Gremlin Database for PMG
resource "azurerm_cosmosdb_gremlin_database" "pmg" {
  name                = "sap_llm_pmg"
  resource_group_name = azurerm_resource_group.cosmos.name
  account_name        = azurerm_cosmosdb_account.main.name

  # Serverless - no throughput setting needed
}

# Gremlin Graph for Documents
resource "azurerm_cosmosdb_gremlin_graph" "documents" {
  name                = "documents"
  resource_group_name = azurerm_resource_group.cosmos.name
  account_name        = azurerm_cosmosdb_account.main.name
  database_name       = azurerm_cosmosdb_gremlin_database.pmg.name

  partition_key_path    = "/doc_id"
  partition_key_version = 2

  # Indexing policy
  index_policy {
    automatic      = true
    indexing_mode  = "consistent"

    included_path {
      path = "/*"
    }

    excluded_path {
      path = "/\"_etag\"/?"
    }
  }

  # Unique keys
  unique_key {
    paths = ["/doc_id", "/version_hash"]
  }
}

# Virtual Network for Cosmos DB
resource "azurerm_virtual_network" "cosmos" {
  name                = "sap-llm-${var.environment}-cosmos-vnet"
  location            = azurerm_resource_group.cosmos.location
  resource_group_name = azurerm_resource_group.cosmos.name
  address_space       = ["10.1.0.0/16"]

  tags = var.tags
}

resource "azurerm_subnet" "cosmos" {
  name                 = "cosmos-subnet"
  resource_group_name  = azurerm_resource_group.cosmos.name
  virtual_network_name = azurerm_virtual_network.cosmos.name
  address_prefixes     = ["10.1.1.0/24"]

  service_endpoints = ["Microsoft.AzureCosmosDB"]
}

# Variables
variable "environment" {
  type = string
}

variable "location" {
  type = string
}

variable "tags" {
  type = map(string)
}

# Outputs
output "endpoint" {
  value = azurerm_cosmosdb_account.main.endpoint
}

output "primary_key" {
  value     = azurerm_cosmosdb_account.main.primary_key
  sensitive = true
}

output "connection_strings" {
  value     = azurerm_cosmosdb_account.main.connection_strings
  sensitive = true
}
