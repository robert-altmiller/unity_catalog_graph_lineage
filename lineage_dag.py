# Databricks notebook source
# DBTITLE 1,Pip Install Libraries
# MAGIC %pip install networkx

# COMMAND ----------

# DBTITLE 1,Restart Python Environment
dbutils.library.restartPython()

# COMMAND ----------

# DBTITLE 1,:Library Imports
import pandas as pd
import numpy as np
import pyspark.sql.functions as F
import networkx as nx
import plotly.graph_objects as go
from IPython.display import display, HTML
from scipy.spatial import KDTree

# COMMAND ----------

# DBTITLE 1,Get Column Lineage Data For Current Year
df_col_lineage = spark.sql(f"""
  SELECT
      account_id, metastore_id, workspace_id, 
      COALESCE(entity_type, 'Standalone') AS entity_type, -- Replace NULL with "Standalone"
      count(DISTINCT entity_id) AS unique_entity_id_count,
      source_table_full_name, source_table_catalog, source_table_schema, source_table_name, source_column_name,
      count(DISTINCT source_path) AS unique_source_path_count, 
      SPLIT_PART(source_path, '/', 3) AS s3_bucket_name, 
      source_type, target_table_full_name, target_table_catalog, 
      target_table_schema, target_table_name, target_column_name, 
      target_path, target_type, created_by, 
      MAX(event_date) AS max_event_date
  FROM system.access.column_lineage
  WHERE source_table_catalog <> target_table_catalog AND event_date >= MAKE_DATE(YEAR(CURRENT_DATE()), 1, 1) -- For current data
  GROUP BY
      account_id, metastore_id, workspace_id, entity_type,
      source_table_full_name, source_table_catalog, source_table_schema, source_table_name, source_column_name,
      SPLIT_PART(source_path, '/', 3), source_type, target_table_full_name, 
      target_table_catalog, target_table_schema, target_table_name,
      target_column_name, target_path, target_type, created_by
""")
# Trim column names (remove extra spaces) and break apart Dataframes
df_col_lineage = df_col_lineage.select([col.strip() for col in df_col_lineage.columns])


# Get column lineage data at a specific graph level
def get_column_lineage_data(source_catalog_list, source_schema_list, source_table_list, data_level):
    column_lineage_data_dict = {
        "catalog": df_col_lineage.where(F.col("source_table_catalog").isin(source_catalog_list)),
        "schema": df_col_lineage.where(F.concat(F.col("source_table_catalog"), F.lit("."), F.col("source_table_schema")).isin(source_schema_list)),
        "table": df_col_lineage.where(F.col("source_table_full_name").isin(source_table_list)),
        "column": df_col_lineage.where(F.col("source_table_full_name").isin(source_table_list))
    }
    if data_level not in column_lineage_data_dict:
        raise ValueError(f"Invalid data_level: {data_level}. Expected one of {list(column_lineage_data_dict.keys())}")
    # Select and group by all columns except source_column_name and target_column_name
    grouped_df = column_lineage_data_dict[data_level].groupBy(
        "source_table_catalog", "source_table_schema", "source_table_full_name",
        "target_table_catalog", "target_table_schema", "target_table_full_name",
        "entity_type", "target_type"
    ).agg(
        F.collect_set("source_column_name").alias("source_column_list"),
        F.collect_set("target_column_name").alias("target_column_list"))
    return grouped_df.toPandas().applymap(lambda x: tuple(x) if isinstance(x, np.ndarray) else x)

# COMMAND ----------

class LineageGraph:
    def __init__(self, df, src_catalogs, src_schemas, src_tables, data_level, graph_level):
        """Initialize the lineage graph with a dataframe, graph level, and title."""
        self.df = df
        self.metadata = {"catalog": src_catalogs, "schema": src_schemas, "table": src_tables, "column": src_tables}
        self.data_level = data_level
        self.level = graph_level
        self.node_hover_text = {} # For keeping tracking of source and target columns
        self.G = nx.DiGraph()  # Directed graph for lineage relationships


    def get_dynamic_title(self):
        """Create a dynamic title based on the graph level and data level."""
        if self.data_level == "column": self.data_level = "table" # Need to back up one level for title to be correct
        return f"Target {self.level.capitalize()}s Updated by Source {self.data_level.capitalize()}s: {self.metadata[self.data_level]}"


    def adjust_positions(self, pos, min_distance):
        """Optimize node spacing using KD-Tree to avoid overlap."""
        if not pos or len(pos) < 2:  # Skip adjustment if there are fewer than two nodes
            return pos  

        nodes = list(pos.keys())
        points = np.array([pos[node] for node in nodes])

        if len(points) < 2 or points.shape[1] != 2:  # Ensure valid 2D points
            return pos  

        tree = KDTree(points)  # Build KD-Tree for fast spatial lookup
        for i, node in enumerate(nodes):
            indices = tree.query_ball_point(points[i], min_distance)
            for j in indices:
                if i != j:
                    x1, y1 = points[i]
                    x2, y2 = points[j]
                    dist = np.sqrt((x2 - x1) ** 2 + (y2 - y1) ** 2)
                    if dist < min_distance and dist > 0:  # Adjust positions if too close
                        shift_x = (x2 - x1) / dist * min_distance
                        shift_y = (y2 - y1) / dist * min_distance
                        points[j] = (x2 + shift_x, y2 + shift_y)
        return {node: tuple(points[i]) for i, node in enumerate(nodes)}


    def get_label(self, table_name, entity, type_label, column_names=None):
        """Generate a node label based on the graph level (catalog, schema, table, column)."""
        if table_name is None: 
            return None
        parts = (table_name.split('.') + ["N/A"] * 3)[:3]  # Ensure at least three components
        labels = {
            "catalog": f"Catalog: {parts[0]}",
            "schema": f"Catalog: {parts[0]}<br>Schema: {parts[1]}",
            "table": f"Catalog: {parts[0]}<br>Schema: {parts[1]}<br>Table: {parts[2]}",
            "column": f"Catalog: {parts[0]}<br>Schema: {parts[1]}<br>Table: {parts[2]}"
        }
        if self.level in ["catalog", "schema"]:
            return f"{labels[self.level]}<br>(<b>{type_label}</b>)"
        return f"{labels[self.level]} <br>(<b>{type_label}: {entity}</b>)"


    def update_node_hover_text(self, label, column_list):
        """Update the hover text for a given node label with formatted column list."""
        if label:
            hover_text = label
            if not pd.isna(column_list):
                hover_text_before = (hover_text.split("(")[0]).strip()
                hover_text_after = (f"({hover_text.split('(')[1]}").strip()
                hover_text = hover_text_before + f"Columns: {', '.join(column_list)}<br>" + hover_text_after
            self.node_hover_text[label] = hover_text


    def build_graph(self):
        """Build the directed lineage graph from the dataframe."""
        edges = self.df.dropna(subset=["source_table_full_name", "entity_type"])

        for _, row in edges.iterrows():
            source_cols = row.get("source_column_list", None)
            target_cols = row.get("target_column_list", None)

            source_label = self.get_label(row["source_table_full_name"], row["entity_type"], "Source", source_cols)
            target_label = None if pd.isna(row["target_table_full_name"]) else self.get_label(row["target_table_full_name"], row["target_type"], "Target", target_cols)

            # Update and format node hover text for columns
            self.update_node_hover_text(source_label, row["source_column_list"])
            self.update_node_hover_text(target_label, row["target_column_list"])

            if source_label and target_label:
                self.G.add_edge(source_label, target_label)  # Create an edge between nodes
            elif source_label:
                self.G.add_node(source_label)  # Add standalone source node with no edges
            elif target_label:
                self.G.add_node(target_label)  # Add standalone target node with no edges


    def compute_positions(self):
        """Compute optimized positions for graph layout."""
        if not self.G.nodes:  # Return empty if there are no nodes
            return {}

        node_count = len(self.G.nodes)
        if node_count == 1:  # Single node case
            return {list(self.G.nodes())[0]: (0, 0)}

        if node_count <= 30:
            pos = nx.spring_layout(self.G, seed=42, k=0.8)  # Use force-directed layout
            return self.adjust_positions(pos, min_distance=0.3)
        else:
            k_dynamic = max(0.3, min(1.5, 1.2 / np.log10(node_count + 10)))  # Dynamic spacing
            pos = nx.spring_layout(self.G, seed=42, k=k_dynamic)
            return self.adjust_positions(pos, min_distance=k_dynamic * 1.5)


    def create_plot(self):
        """Generate a Plotly visualization of the lineage graph with total counts."""
        pos = self.compute_positions()

        # Compute target counts
        target_column = {
            "catalog": "target_table_catalog",
            "schema": "target_table_schema",
            "table": "target_table_full_name",
            "column": "target_column_list"
        }

        # Ensure correct counting of unique hierarchical entities
        if self.level == "catalog":
            total_targets = self.df[target_column["catalog"]].nunique()
        elif self.level == "schema":
            total_targets = self.df.drop_duplicates(subset=[target_column["catalog"], target_column["schema"]]).shape[0]
        elif self.level == "table":
            total_targets = self.df.drop_duplicates(subset=[target_column["table"]]).shape[0]
        elif self.level == "column":
            # Counts the number of unique column names appearing anywhere in the target_column_list column across all rows
            df_target_cols = self.df.drop_duplicates(subset=[target_column["column"]])
            total_targets = len(set(column for columns in df_target_cols["target_column_list"] for column in columns))
        else:
            total_targets = 0  # Default fallback

        # Create edge traces
        edge_x, edge_y = [], []
        for edge in self.G.edges():
            x0, y0 = pos[edge[0]]
            x1, y1 = pos[edge[1]]
            edge_x.extend([x0, x1, None])
            edge_y.extend([y0, y1, None])

        edge_trace = go.Scatter(
            x=edge_x, y=edge_y,
            line=dict(width=1.5, color='gray'),
            hoverinfo='none',
            mode='lines',
            name="Edges"
        )

        # Create node traces
        node_x, node_y, node_text, hover_text = [], [], [], []
        for node in self.G.nodes():
            x, y = pos[node]
            node_x.append(x)
            node_y.append(y)
            node_text.append(node)
            hover_text.append(self.node_hover_text.get(node, node))  # Hover info

        if self.level != "column": hover_text = node_text
        node_trace = go.Scatter(
            x=node_x, y=node_y,
            mode='markers+text',
            marker=dict(size=14, color='blue', opacity=0.8),
            text=node_text,
            textposition=["bottom center" if i % 2 == 0 else "top center" for i in range(len(node_text))],
            hoverinfo='text',
            hovertext=hover_text,  # Only show columns on hover
            name="Nodes"
        )

        # Configure Plotly figure layout with annotation
        fig = go.Figure(data=[edge_trace, node_trace])
        fig.update_layout(
            title=self.get_dynamic_title(),
            title_x=0.5,
            width=None,
            height=900,
            margin=dict(l=50, r=50, t=50, b=50),
            hovermode='closest',
            dragmode="pan",
            annotations=[
                dict(
                    x=0.99, y=1.05,  # Adjust position near the top right
                    xref="paper", yref="paper",
                    text=f"Total Target {self.level.capitalize()}s: {total_targets}",
                    showarrow=False,
                    font=dict(size=14, color="black")
                )
            ]
        )
        return fig


    def show_graph(self):
        """Display the lineage graph with a dynamic resizing script for Plotly."""
        js_script = """
        <script>
            function updatePlotlySize() {
                let fig = document.getElementsByClassName('plotly-graph-div')[0];
                if (fig) {
                    fig.style.width = window.innerWidth * 0.9 + 'px';
                }
            }
            window.addEventListener('resize', updatePlotlySize);
            window.onload = updatePlotlySize;
        </script>
        """
        display(HTML(js_script))  # Inject resizing script for responsiveness
        fig = self.create_plot()
        fig.show()

# COMMAND ----------

# MAGIC %md
# MAGIC ## Build Graph Examples (Catalog Level)
# MAGIC 1. What are __ALL__ the 'target' catalogs that get updated for >= 1 'source' catalog?
# MAGIC     - data_level = "catalog" and graph_level = "catalog"
# MAGIC 2. What __SUBSET__ of 'target' catalogs get updated for >= 1 'source' catalog.schema?
# MAGIC     - data_level = "schema" and graph_level = "catalog"
# MAGIC 3. What __SUBSET__ of 'target' catalogs get updated for >= 1 'source' catalog.schema.table? 
# MAGIC     - data_level = "table" and graph_level = "catalog"
# MAGIC
# MAGIC ## Build Graph Examples (Schema Level)
# MAGIC 1. What are __ALL__ the 'target' schemas that get updated for >= 1 'source' catalog? 
# MAGIC     - data_level = "catalog" and graph_level = "schema"
# MAGIC 2. What __SUBSET__ of 'target' schemas get updated for >= 1 'source' catalog.schema?: 
# MAGIC     - data_level = "schema" and graph_level = "schema"
# MAGIC 3. What __SUBSET__ of 'target' schemas get updated for >= 1 'source' catalog.schema.table? 
# MAGIC     - data_level = "table" and graph_level = "schema"
# MAGIC
# MAGIC ## Build Graph Examples (Table Level)
# MAGIC 1. What are __ALL__ the 'target' tables that get updated for >= 1 'source' catalog? 
# MAGIC     - data_level = "catalog" and graph_level = "table"
# MAGIC 2. What __SUBSET__ of target 'tables' get updated for >= 1 'source' catalog.schema?: 
# MAGIC     - data_level = "schema" and graph_level = "table"
# MAGIC 3. What __SUBSET__ of 'target' schemas get updated for >= 1 'source' catalog.schema.table? 
# MAGIC     - data_level = "table" and graph_level = "table"
# MAGIC
# MAGIC ## Build Graph Examples (Column Level)
# MAGIC 1. What are __ALL__ the 'target' columns that get updated for >= 1 'source' catalog? 
# MAGIC     - data_level = "catalog" and graph_level = "column"
# MAGIC 2. What __SUBSET__ of 'target' columns that get updated for >= 1 'source' catalog.schema?: 
# MAGIC     - data_level = "schema" and graph_level = "column"
# MAGIC 3. What __SUBSET__ of 'target' columns that get updated for >= 1 'source' catalog.schema.table? 
# MAGIC     - data_level = "table" and graph_level = "column"

# COMMAND ----------

# # Graph parameters
# n = 50
# src_catalogs_all = spark.sql("""SELECT catalog_name FROM system.information_schema.catalogs""")
# src_catalog = src_catalogs_all.select("catalog_name").toPandas()["catalog_name"].tolist()[:n]
# src_catalog.remove("caio")
# print(src_catalog)

# COMMAND ----------

# DBTITLE 1,Create and Show Lineage Graph With Nodes and Edges
# Graph parameters
src_catalog_list = ["my_catalog"] #change
src_schema_list = ["system.billing"] #change
src_table_list = ['system.billing.list_prices'] #change

# Graph and Data levels
data_level = "column" # 'catalog' or 'schema' or 'table' or 'column'
graph_level = "column" # 'catalog' or 'schema' or 'table' or 'column'

lineage_data = get_column_lineage_data(src_catalog_list, src_schema_list, src_table_list, data_level)
lineage_graph = LineageGraph(lineage_data, src_catalog_list, src_schema_list, src_table_list, data_level, graph_level)
lineage_graph.build_graph()
lineage_graph.show_graph()

# COMMAND ----------