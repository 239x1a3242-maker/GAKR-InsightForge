"""File-based database utilities for JSON storage."""
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional
from datetime import datetime
import uuid

# Base data directory
DATA_DIR = Path(__file__).parent.parent / "data"
UPLOADS_DIR = Path(__file__).parent.parent / "uploads"

# Create directories if they don't exist
DATA_DIR.mkdir(exist_ok=True)
UPLOADS_DIR.mkdir(exist_ok=True)
(DATA_DIR / "datasets").mkdir(exist_ok=True)
(UPLOADS_DIR / "datasets").mkdir(exist_ok=True)


class FileDB:
    """Simple file-based JSON database."""

    @staticmethod
    def load(filepath: Path) -> Dict[str, Any]:
        """Load JSON file."""
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    return json.load(f)
        except (json.JSONDecodeError, IOError):
            pass
        return {}

    @staticmethod
    def save(filepath: Path, data: Dict[str, Any]) -> None:
        """Save to JSON file."""
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def append_list(filepath: Path, item: Dict[str, Any]) -> None:
        """Append item to JSON array file."""
        data = FileDB.load_list(filepath)
        data.append(item)
        filepath.parent.mkdir(parents=True, exist_ok=True)
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(data, f, indent=2, default=str)

    @staticmethod
    def load_list(filepath: Path) -> List[Dict[str, Any]]:
        """Load JSON array file."""
        try:
            if filepath.exists():
                with open(filepath, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return data if isinstance(data, list) else []
        except (json.JSONDecodeError, IOError):
            pass
        return []


# User Management
class UserDB:
    """User database management."""
    
    USERS_FILE = DATA_DIR / "users.json"

    @classmethod
    def get_all_users(cls) -> List[Dict[str, Any]]:
        """Get all users."""
        return FileDB.load_list(cls.USERS_FILE)

    @classmethod
    def get_by_email(cls, email: str) -> Optional[Dict[str, Any]]:
        """Get user by email."""
        users = cls.get_all_users()
        return next((u for u in users if u.get("email") == email), None)

    @classmethod
    def get_by_id(cls, user_id: str) -> Optional[Dict[str, Any]]:
        """Get user by ID."""
        users = cls.get_all_users()
        return next((u for u in users if u.get("id") == user_id), None)

    @classmethod
    def create(cls, email: str, password_hash: str, full_name: str = "") -> Dict[str, Any]:
        """Create new user."""
        if cls.get_by_email(email):
            raise ValueError(f"User with email {email} already exists")
        
        user = {
            "id": str(uuid.uuid4()),
            "email": email,
            "password_hash": password_hash,
            "full_name": full_name,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        FileDB.append_list(cls.USERS_FILE, user)
        return user

    @classmethod
    def update(cls, user_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update user."""
        users = cls.get_all_users()
        for i, user in enumerate(users):
            if user.get("id") == user_id:
                user.update(kwargs)
                user["updated_at"] = datetime.utcnow().isoformat()
                FileDB.save(cls.USERS_FILE, users)
                return user
        return None

    @classmethod
    def delete(cls, user_id: str) -> bool:
        """Delete user."""
        users = cls.get_all_users()
        initial_len = len(users)
        users = [u for u in users if u.get("id") != user_id]
        if len(users) < initial_len:
            FileDB.save(cls.USERS_FILE, users)
            return True
        return False


# Dataset Management
class DatasetDB:
    """Dataset database management."""
    
    DATASETS_INDEX = DATA_DIR / "datasets_index.json"
    DATASETS_DIR = DATA_DIR / "datasets"

    @classmethod
    def _get_dataset_file(cls, dataset_id: str) -> Path:
        """Get dataset file path."""
        return cls.DATASETS_DIR / f"{dataset_id}.json"

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all datasets.

        Handles legacy format where the index file was accidentally saved as
        ``{"datasets": [... ]}`` due to a bug in earlier versions. We
        normalize to a simple list and rewrite the file if necessary so that
        callers always receive a plain list of dataset metadata.
        """
        raw = FileDB.load(cls.DATASETS_INDEX)
        # normalize
        if isinstance(raw, dict):
            # support legacy wrappers
            datasets = raw.get("datasets")
            if isinstance(datasets, list):
                # remove any embedded data rows for size reasons before
                # migrating the file
                cleaned = []
                for d in datasets:
                    if isinstance(d, dict):
                        d = {k: v for k, v in d.items() if k != "data"}
                    cleaned.append(d)
                FileDB.save(cls.DATASETS_INDEX, cleaned)
                return cleaned
            # unexpected structure, return empty list
            return []
        elif isinstance(raw, list):
            return raw
        else:
            return []

    @classmethod
    def get_by_user(cls, user_id: str) -> List[Dict[str, Any]]:
        """Get datasets by user ID."""
        datasets = cls.get_all()
        return [d for d in datasets if d.get("user_id") == user_id]

    @classmethod
    def get_by_id(cls, dataset_id: str) -> Optional[Dict[str, Any]]:
        """Get dataset by ID."""
        dataset_file = cls._get_dataset_file(dataset_id)
        data = FileDB.load(dataset_file)
        if data:
            return data
        return None

    @classmethod
    def create(cls, user_id: str, name: str, filename: str, 
               rows: int = 0, columns: int = 0, file_size: int = 0) -> Dict[str, Any]:
        """Create new dataset record."""
        dataset_id = str(uuid.uuid4())
        
        dataset = {
            "id": dataset_id,
            "user_id": user_id,
            "name": name,
            "filename": filename,
            "rows": rows,
            "columns": columns,
            "file_size": file_size,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        # Save dataset metadata to individual file
        FileDB.save(cls._get_dataset_file(dataset_id), dataset)
        
        # Add to index
        FileDB.append_list(cls.DATASETS_INDEX, dataset)
        
        return dataset

    @classmethod
    def get_data(cls, dataset_id: str) -> Optional[List[Dict[str, Any]]]:
        """Get dataset data rows."""
        dataset_file = cls._get_dataset_file(dataset_id)
        data = FileDB.load(dataset_file)
        return data.get("data", []) if data else None

    @classmethod
    def save_data(cls, dataset_id: str, rows: List[Dict[str, Any]]) -> None:
        """Save dataset data rows."""
        dataset = cls.get_by_id(dataset_id)
        if dataset:
            dataset["data"] = rows
            dataset["rows"] = len(rows)
            dataset["columns"] = len(rows[0]) if rows else 0
            dataset["updated_at"] = datetime.utcnow().isoformat()
            FileDB.save(cls._get_dataset_file(dataset_id), dataset)
            
            # Update index (strip heavy data field)
            index = cls.get_all()
            idx_entry = {k: v for k, v in dataset.items() if k != "data"}
            for i, d in enumerate(index):
                if d.get("id") == dataset_id:
                    index[i] = idx_entry
                    break
            FileDB.save(cls.DATASETS_INDEX, index)

    @classmethod
    def update(cls, dataset_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update dataset metadata."""
        dataset = cls.get_by_id(dataset_id)
        if dataset:
            dataset.update(kwargs)
            dataset["updated_at"] = datetime.utcnow().isoformat()
            FileDB.save(cls._get_dataset_file(dataset_id), dataset)
            
            # Update index (strip out data rows)
            index = cls.get_all()
            idx_entry = {k: v for k, v in dataset.items() if k != "data"}
            for i, d in enumerate(index):
                if d.get("id") == dataset_id:
                    index[i] = idx_entry
                    break
            FileDB.save(cls.DATASETS_INDEX, index)
            
            return dataset
        return None

    @classmethod
    def delete(cls, dataset_id: str) -> bool:
        """Delete dataset."""
        dataset_file = cls._get_dataset_file(dataset_id)
        if dataset_file.exists():
            dataset_file.unlink()
        
        index = cls.get_all()
        initial_len = len(index)
        index = [d for d in index if d.get("id") != dataset_id]
        if len(index) < initial_len:
            FileDB.save(cls.DATASETS_INDEX, index)
            return True
        return False


# Model Management
class ModelDB:
    """Model database management."""
    
    MODELS_INDEX = DATA_DIR / "models_index.json"
    MODELS_DIR = DATA_DIR / "models"

    @classmethod
    def _get_model_file(cls, model_id: str) -> Path:
        """Get model file path."""
        return cls.MODELS_DIR / f"{model_id}.json"

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        """Get all models.

        Performs a similar migration to :meth:`DatasetDB.get_all` to handle
        cases where the index was written as ``{"models": [... ]}``.
        """
        raw = FileDB.load(cls.MODELS_INDEX)
        if isinstance(raw, dict):
            models = raw.get("models")
            if isinstance(models, list):
                FileDB.save(cls.MODELS_INDEX, models)
                return models
            return []
        elif isinstance(raw, list):
            return raw
        else:
            return []

    @classmethod
    def get_by_user(cls, user_id: str) -> List[Dict[str, Any]]:
        """Get models by user ID."""
        models = cls.get_all()
        return [m for m in models if m.get("user_id") == user_id]

    @classmethod
    def create(cls, user_id: str, name: str, algorithm: str, 
               dataset_id: str, status: str = "training") -> Dict[str, Any]:
        """Create new model."""
        model_id = str(uuid.uuid4())
        
        model = {
            "id": model_id,
            "user_id": user_id,
            "name": name,
            "algorithm": algorithm,
            "dataset_id": dataset_id,
            "status": status,
            "accuracy": None,
            "metrics": {},
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        
        FileDB.save(cls._get_model_file(model_id), model)
        FileDB.append_list(cls.MODELS_INDEX, model)
        
        return model

    @classmethod
    def update(cls, model_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        """Update model."""
        model_file = cls._get_model_file(model_id)
        model = FileDB.load(model_file)
        if model:
            model.update(kwargs)
            model["updated_at"] = datetime.utcnow().isoformat()
            FileDB.save(model_file, model)
            
            # Update index
            index = cls.get_all()
            for i, m in enumerate(index):
                if m.get("id") == model_id:
                    index[i] = model
                    break
            FileDB.save(cls.MODELS_INDEX, index)
            
            return model
        return None


# Report & Dashboard Management
class ReportDB:
    """Report database management."""

    REPORTS_INDEX = DATA_DIR / "reports_index.json"

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        raw = FileDB.load(cls.REPORTS_INDEX)
        if isinstance(raw, list):
            return raw
        return []

    @classmethod
    def get_by_user(cls, user_id: str) -> List[Dict[str, Any]]:
        return [r for r in cls.get_all() if r.get("user_id") == user_id]

    @classmethod
    def get_by_id(cls, report_id: str) -> Optional[Dict[str, Any]]:
        return next((r for r in cls.get_all() if r.get("id") == report_id), None)

    @classmethod
    def create(cls, user_id: str, name: str, description: str = "", dataset_id: str = "") -> Dict[str, Any]:
        report = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "description": description,
            "dataset_id": dataset_id,
            "pages": [],
            "page_count": 0,
            "is_published": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        FileDB.append_list(cls.REPORTS_INDEX, report)
        return report

    @classmethod
    def update(cls, report_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        reports = cls.get_all()
        for i, r in enumerate(reports):
            if r.get("id") == report_id:
                r.update(kwargs)
                r["updated_at"] = datetime.utcnow().isoformat()
                FileDB.save(cls.REPORTS_INDEX, reports)
                return r
        return None

    @classmethod
    def delete(cls, report_id: str) -> bool:
        reports = cls.get_all()
        new_reports = [r for r in reports if r.get("id") != report_id]
        if len(new_reports) < len(reports):
            FileDB.save(cls.REPORTS_INDEX, new_reports)
            return True
        return False


class DashboardDB:
    """Dashboard database management."""

    DASHBOARDS_INDEX = DATA_DIR / "dashboards_index.json"

    @classmethod
    def get_all(cls) -> List[Dict[str, Any]]:
        raw = FileDB.load(cls.DASHBOARDS_INDEX)
        if isinstance(raw, list):
            return raw
        return []

    @classmethod
    def get_by_user(cls, user_id: str) -> List[Dict[str, Any]]:
        return [d for d in cls.get_all() if d.get("user_id") == user_id]

    @classmethod
    def get_by_id(cls, dashboard_id: str) -> Optional[Dict[str, Any]]:
        return next((d for d in cls.get_all() if d.get("id") == dashboard_id), None)

    @classmethod
    def create(cls, user_id: str, name: str, description: str = "") -> Dict[str, Any]:
        dashboard = {
            "id": str(uuid.uuid4()),
            "user_id": user_id,
            "name": name,
            "description": description,
            "widgets": [],
            "layout": [],
            "is_published": False,
            "created_at": datetime.utcnow().isoformat(),
            "updated_at": datetime.utcnow().isoformat(),
        }
        FileDB.append_list(cls.DASHBOARDS_INDEX, dashboard)
        return dashboard

    @classmethod
    def update(cls, dashboard_id: str, **kwargs) -> Optional[Dict[str, Any]]:
        dashboards = cls.get_all()
        for i, d in enumerate(dashboards):
            if d.get("id") == dashboard_id:
                d.update(kwargs)
                d["updated_at"] = datetime.utcnow().isoformat()
                FileDB.save(cls.DASHBOARDS_INDEX, dashboards)
                return d
        return None

    @classmethod
    def delete(cls, dashboard_id: str) -> bool:
        dashboards = cls.get_all()
        new_dashboards = [d for d in dashboards if d.get("id") != dashboard_id]
        if len(new_dashboards) < len(dashboards):
            FileDB.save(cls.DASHBOARDS_INDEX, new_dashboards)
            return True
        return False


# Initialize default data
def init_db():
    """Initialize database with default files."""
    # Create empty users list if not exists
    if not UserDB.USERS_FILE.exists():
        FileDB.save(UserDB.USERS_FILE, [])
    
    # Create empty datasets index
    if not DatasetDB.DATASETS_INDEX.exists():
        FileDB.save(DatasetDB.DATASETS_INDEX, [])
    
    # Create empty models index
    if not ModelDB.MODELS_INDEX.exists():
        FileDB.save(ModelDB.MODELS_INDEX, [])

    # Create empty reports index
    if not ReportDB.REPORTS_INDEX.exists():
        FileDB.save(ReportDB.REPORTS_INDEX, [])

    # Create empty dashboards index
    if not DashboardDB.DASHBOARDS_INDEX.exists():
        FileDB.save(DashboardDB.DASHBOARDS_INDEX, [])

    print("✓ File-based database initialized")
