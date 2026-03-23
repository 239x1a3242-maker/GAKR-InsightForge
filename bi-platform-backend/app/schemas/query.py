"""Query engine schemas."""
from pydantic import BaseModel, Field
from typing import Optional, List, Dict, Any, Union
from enum import Enum


class AggregationFunction(str, Enum):
    """Aggregation functions."""
    SUM = "sum"
    AVG = "avg"
    MIN = "min"
    MAX = "max"
    COUNT = "count"
    COUNT_DISTINCT = "count_distinct"
    STDEV = "stdev"
    VAR = "var"
    MEDIAN = "median"


class FilterOperator(str, Enum):
    """Filter operators."""
    EQ = "eq"
    NEQ = "neq"
    GT = "gt"
    GTE = "gte"
    LT = "lt"
    LTE = "lte"
    IN = "in"
    NOT_IN = "not_in"
    CONTAINS = "contains"
    STARTS_WITH = "starts_with"
    ENDS_WITH = "ends_with"
    IS_NULL = "is_null"
    IS_NOT_NULL = "is_not_null"
    BETWEEN = "between"


class SortDirection(str, Enum):
    """Sort directions."""
    ASC = "asc"
    DESC = "desc"


class Filter(BaseModel):
    """Filter condition."""
    field: str
    operator: FilterOperator
    value: Optional[Union[str, int, float, bool, List[Any]]] = None


class Aggregation(BaseModel):
    """Aggregation definition."""
    field: str
    function: AggregationFunction
    alias: Optional[str] = None


class Sort(BaseModel):
    """Sort definition."""
    field: str
    direction: SortDirection = SortDirection.ASC


class QueryRequest(BaseModel):
    """Query request schema."""
    dataset_id: str
    table_name: Optional[str] = None
    
    # Fields to select
    fields: Optional[List[str]] = None
    
    # Group by
    group_by: Optional[List[str]] = None
    
    # Aggregations
    aggregations: Optional[List[Aggregation]] = None
    
    # Filters
    filters: Optional[List[Filter]] = None
    
    # Sorting
    sort: Optional[List[Sort]] = None
    
    # Pagination
    page: int = Field(1, ge=1)
    page_size: int = Field(1000, ge=1, le=100000)
    
    # Limits
    limit: Optional[int] = Field(None, ge=1, le=1000000)


class QueryResponse(BaseModel):
    """Query response schema."""
    columns: List[Dict[str, Any]]
    data: List[Dict[str, Any]]
    total_rows: int
    page: int
    page_size: int
    execution_time_ms: int


class DrillThroughRequest(BaseModel):
    """Drill-through request schema."""
    dataset_id: str
    source_visual_id: str
    target_page_id: str
    filters: List[Filter]
    selected_values: Dict[str, Any]


class CrossFilterRequest(BaseModel):
    """Cross-filter request schema."""
    dataset_id: str
    source_visual_id: str
    target_visual_ids: List[str]
    filters: List[Filter]


class TimeIntelligenceRequest(BaseModel):
    """Time intelligence request schema."""
    dataset_id: str
    date_column: str
    measure: str
    comparison_type: str  # yoy, mom, qoq, ytd, mtd, etc.
    current_period: Optional[Dict[str, str]] = None  # start_date, end_date


class TimeIntelligenceResponse(BaseModel):
    """Time intelligence response schema."""
    current_value: float
    previous_value: float
    change_amount: float
    change_percent: float
    trend: str  # up, down, flat
