# Unit Catalog (UC) Lineage Graph For Catalogs, Schemas, Tables, and Columns

The Databricks Unity Catalog Graph Lineage Tool is crucial because it enables comprehensive data lineage tracking across multiple levels (catalog, schema, table, and columns).  As a data product owner you will get insights how your data products are being used in other data domains,
business units, and organizations across your Enterprise.  This tool provides full data lineage visibility and helps in understanding dependencies between datasets across catalogs, schemas, tables, and columns.  It's useful for impact analysis—understanding what downstream 
datasets might be affected by upstream changes.  Enterprise teams can quickly evaluate downstream impacts prior to making changes to a dataset, and fix issues before they impact business decisions.  The visual graph supports governance and compliance by giving a holistic view of 
what datasets are being used and where for the entire Databricks regional Metastore, and facilitates access control and security by helping to track sensitive data (e.g., PII, financial records) at the column level.

#### Link to Github Repo: https://github.com/robert-altmiller/unity_catalog_graph_lineage

## Features of the UC Lineage Graph

The Databricks Unity Catalog Graph Lineage Tool has the following features:

- The entire lineage tool runs in a single Databricks Notbook with no complicated app server setup.
- Takes advantage of the Databricks system tables (e.g. column_lineage) which are available with Unity Catalog.
- Tracks how data flows from source to target at multiple granularities.
- Helps data stewards and governance teams ensure data is used and updated correctly.
- Prevents data silos by offering a centralized view of how data is transformed.
- Provides a transparent way to quickly audit data movement, and this critical for regulatory compliance (e.g., GDPR, HIPAA, SOC2).
- Improves data security monitoring by enabling the identication of unexpected data movement.
- If a user resizes their browser window the LineageGraph Python class adjusts the graph width dynamically.

## How many different types of UC Lineage Graphs can be created?

- In the Databricks notebook section __'Create and Show Lineage Graph With Nodes and Edges'__ update the following sections below in _red_, and use the __'Build Graph Examples'__ below to understand the different ways to set __'data_level'__ and __'graph_level'__.  Please update the  __'src_catalog_list'__, __'src_schema_list'__, and __'src_table_list'__ so the get_column_lineage_data() Python function knows what data to query for the UC Lineage Graph.  You can include multiple catalogs, schemas, and tables in each of these lists.

![update_data_graph_level.png](/readme_images/update_data_graph_level.png)

# Build Graph Examples

## Catalog Level
| Description | Data Level | Graph Level |
|------------|-----------|-------------|
| What are **ALL** the 'target' catalogs that get updated for >= 1 'source' catalog? | `'catalog'` | `'catalog'` |
| What **SUBSET** of 'target' catalogs get updated for >= 1 'source' catalog.schema? | `'schema'` | `'catalog'` |
| What **SUBSET** of 'target' catalogs get updated for >= 1 'source' catalog.schema.table? | `'table'` | `'catalog'` |

## Schema Level
| Description | Data Level | Graph Level |
|------------|-----------|-------------|
| What are **ALL** the 'target' schemas that get updated for >= 1 'source' catalog? | `'catalog'` | `'schema'` |
| What **SUBSET** of 'target' schemas get updated for >= 1 'source' catalog.schema? | `'schema'` | `'schema'` |
| What **SUBSET** of 'target' schemas get updated for >= 1 'source' catalog.schema.table? | `'table'` | `'schema'` |

## Table Level
| Description | Data Level | Graph Level |
|------------|-----------|-------------|
| What are **ALL** the 'target' tables that get updated for >= 1 'source' catalog? | `'catalog'` | `'table'` |
| What **SUBSET** of 'target' tables get updated for >= 1 'source' catalog.schema? | `'schema'` | `'table'` |
| What **SUBSET** of 'target' tables get updated for >= 1 'source' catalog.schema.table? | `'table'` | `'table` |

## Column Level
| Description | Data Level | Graph Level |
|------------|-----------|-------------|
| What are **ALL** the 'target' columns that get updated for >= 1 'source' catalog? | `'catalog'` | `'column'` |
| What **SUBSET** of 'target' columns get updated for >= 1 'source' catalog.schema? | `'schema'` | `'column'` |
| What **SUBSET** of 'target' columns get updated for >= 1 'source' catalog.schema.table? | `'table'` | `'column'` |

## Show different UC Lineage Graphs examples

- Here is a 'source catalog' to 'target catalog' UC Lineage Graph example at the 'catalog' data level.  This shows all the catalogs for the entire Databricks Metastore which are updated
with table data in the 'system' Databricks catalog.  This can give you a holistic understanding of high-level data movement between catalogs for a single or multiple source catalog.

- data_level = "catalog" and graph_level = "catalog"

![catalog_to_catalog__catalog_level_graph.png](/readme_images/catalog_to_catalog__catalog_level_graph.png)

- Here is a 'source catalog' to 'target catalog' UC Lineage Graph example at the 'schema' data level.  This shows a subset the Metastore catalogs which are updated
with table data in the 'system.billing' Databricks schema.  This can give you a holistic understanding of high-level data movement between catalogs for a single or multiple source schemas.

- data_level = "schema" and graph_level = "catalog"

![catalog_to_catalog__schema_level_graph.png](/readme_images/catalog_to_catalog__schema_level_graph.png)

- Here is a 'source catalog' to 'target catalog' UC Lineage Graph example at the 'table' data level.  This shows a subset the Metastore catalogs which are updated
with table data in the 'system.billing.list_prices' Databricks table.  This can give you a holistic understanding of high-level data movement between catalogs for a single or multiple source tables.

- data_level = "table" and graph_level = "catalog"

![catalog_to_catalog__table_level_graph.png](/readme_images/catalog_to_catalog__table_level_graph.png)


- Here is a 'source schema' to 'target schema' UC Lineage Graph example at the 'schema' data level.  This shows a subset the Metastore schemas which are updated
with table data in the 'system.billing' Databricks schema.  This can give you a holistic understanding of high-level data movement between schemas for a single or multiple source schemas.

- data_level = "schema" and graph_level = "schema"

![schema_to_schema__schema_level_graph.png](/readme_images/schema_to_schema__schema_level_graph.png)

- Here is a 'source table' to 'target table' UC Lineage Graph example at the 'table' data level.  This shows a subset the Metastore tables which are updated
with table data in the 'system.billing.list_prices' Databricks table.  This can give you a holistic understanding of high-level data movement between tables for a single or multiple source tables.

- data_level = "table" and graph_level = "table"

![table_to_table__table_level_graph.png](/readme_images/table_to_table__table_level_graph.png)

- Here is a 'source table (with columns)' to 'target table (with columns)' UC Lineage Graph example at the 'column' data level.  This shows a subset the Metastore tables + columns which are updated
with table data in the 'system.billing.list_prices' Databricks table.  This can give you a holistic understanding of high-level data movement between tables + columns for a single or multiple source tables.

- data_level = "table" and graph_level = "column"

![tblcol_to_tbcol__tablecol_level_graph.png](/readme_images/tblcol_to_tbcol__tablecol_level_graph.png)