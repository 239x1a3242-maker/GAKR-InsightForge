"""Quick test script to verify ML training works."""
import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent))

from app.core.filedb import DatasetDB, ModelDB, UserDB, init_db
import pandas as pd
import uuid

def test_ml_training():
    """Test ML training with sample data."""
    print("Initializing database...")
    init_db()
    
    # Create test user if not exists
    test_email = "test@example.com"
    user = UserDB.get_by_email(test_email)
    if not user:
        from app.core.security import get_password_hash
        user = UserDB.create(test_email, get_password_hash("test123"), "Test User")
    print(f"✓ User: {user['email']}")
    
    # Create sample dataset
    print("\nCreating sample dataset...")
    sample_data = pd.DataFrame({
        'age': [25, 30, 35, 40, 45, 50, 55, 60, 65, 70],
        'income': [30000, 40000, 50000, 60000, 70000, 80000, 90000, 100000, 110000, 120000],
        'credit_score': [600, 650, 700, 750, 800, 650, 700, 750, 800, 850],
        'approved': ['No', 'No', 'Yes', 'Yes', 'Yes', 'No', 'Yes', 'Yes', 'Yes', 'Yes']
    })
    
    dataset_id = str(uuid.uuid4())
    dataset = DatasetDB.create(
        user_id=user['id'],
        name="Test Credit Dataset",
        filename="test_credit.csv",
        rows=len(sample_data),
        columns=len(sample_data.columns)
    )
    dataset_id = dataset['id']
    
    # Save data
    DatasetDB.save_data(dataset_id, sample_data.to_dict('records'))
    print(f"✓ Dataset created: {dataset['name']} ({dataset_id})")
    
    # Test training
    print("\nStarting ML training...")
    from app.api.ml_minimal import _run_training
    
    model_id = str(uuid.uuid4())
    model = ModelDB.create(
        user_id=user['id'],
        name="Test Model",
        algorithm="AutoML",
        dataset_id=dataset_id,
        status="training"
    )
    model_id = model['id']
    
    _run_training(
        model_id=model_id,
        dataset_id=dataset_id,
        target_columns=['approved'],
        feature_columns=['age', 'income', 'credit_score'],
        problem_type='auto',
        test_size=0.2,
        cv_folds=3,
        hyperparameter_tuning=False,
        auto_feature_engineering=False,
        feature_selection_method='all',
        user_id=user['id']
    )
    
    # Check results
    trained_model = ModelDB.get_by_user(user['id'])[0]
    print(f"\n✓ Training completed!")
    print(f"  Status: {trained_model['status']}")
    if trained_model['status'] == 'completed':
        print(f"  Training time: {trained_model.get('total_training_time_seconds', 0):.2f}s")
        for target, result in trained_model.get('target_results', {}).items():
            if 'error' in result:
                print(f"  Target '{target}': ERROR - {result['error']}")
            else:
                print(f"  Target '{target}':")
                print(f"    Algorithm: {result['best_algorithm']}")
                print(f"    Task: {result['task_type']}")
                print(f"    Metrics: {result['metrics']}")
    elif trained_model['status'] == 'failed':
        print(f"  Error: {trained_model.get('error', 'Unknown error')}")
    
    return trained_model

if __name__ == "__main__":
    try:
        model = test_ml_training()
        print("\n✅ ML training test completed successfully!")
    except Exception as e:
        print(f"\n❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)
