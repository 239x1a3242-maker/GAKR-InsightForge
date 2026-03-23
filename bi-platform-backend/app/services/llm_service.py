"""LLM service for AI-generated insights."""
import os
import torch
from typing import Dict, Any, List
from transformers import AutoTokenizer, AutoModelForCausalLM, pipeline
from app.core.config import settings


class LLMService:
    """LLM service for generating insights."""
    
    _instance = None
    _tokenizer = None
    _model = None
    _pipeline = None
    
    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance
    
    def __init__(self):
        if self._tokenizer is None:
            self._load_model()
    
    def _load_model(self):
        """Load the language model."""
        try:
            model_name = settings.LLM_MODEL_NAME
            
            # Load tokenizer
            self._tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Load model with CPU optimization
            self._model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float32,
                device_map="cpu",
                low_cpu_mem_usage=True,
            )
            
            # Create pipeline
            self._pipeline = pipeline(
                "text-generation",
                model=self._model,
                tokenizer=self._tokenizer,
                max_new_tokens=settings.LLM_MAX_TOKENS,
                temperature=settings.LLM_TEMPERATURE,
                do_sample=True,
                pad_token_id=self._tokenizer.eos_token_id,
            )
            
        except Exception as e:
            print(f"Failed to load LLM: {e}")
            self._pipeline = None
    
    def generate_insight(
        self,
        analysis_type: str,
        data_summary: Dict[str, Any],
        context: str = ""
    ) -> Dict[str, Any]:
        """Generate AI insight from analysis data."""
        if self._pipeline is None:
            return {
                'insight': 'LLM not available. Using statistical summary instead.',
                'confidence': 0.0,
            }
        
        # Build prompt
        prompt = self._build_prompt(analysis_type, data_summary, context)
        
        try:
            # Generate response
            response = self._pipeline(prompt)[0]['generated_text']
            
            # Extract the generated part (after the prompt)
            generated = response[len(prompt):].strip()
            
            # Truncate if too long
            if len(generated) > 1000:
                generated = generated[:1000] + "..."
            
            return {
                'insight': generated,
                'confidence': 0.75,
                'tokens_used': len(self._tokenizer.encode(generated)),
            }
            
        except Exception as e:
            return {
                'insight': f'Error generating insight: {str(e)}',
                'confidence': 0.0,
            }
    
    def _build_prompt(
        self,
        analysis_type: str,
        data_summary: Dict[str, Any],
        context: str
    ) -> str:
        """Build prompt for the LLM."""
        
        if analysis_type == 'descriptive':
            prompt = f"""You are a data analyst. Summarize this dataset in 2-3 sentences:

Dataset Overview:
- Rows: {data_summary.get('total_rows', 'N/A')}
- Columns: {data_summary.get('total_columns', 'N/A')}
- Numeric columns: {data_summary.get('numeric_columns', 'N/A')}
- Missing values: {data_summary.get('missing_summary', {}).get('missing_percentage', 'N/A'):.1f}%

Key Insights: {', '.join(data_summary.get('insights', [])[:3])}

Provide a concise executive summary:
"""
        
        elif analysis_type == 'diagnostic':
            prompt = f"""You are a data analyst. Explain the diagnostic findings:

Target Variable: {data_summary.get('target_column', 'N/A')}
Top Correlated Features: {list(data_summary.get('feature_importance', {}).keys())[:3] if data_summary.get('feature_importance') else 'N/A'}
Outliers Detected: {data_summary.get('outliers', {}).get('total_outliers', 'N/A')}

Explain what these findings suggest:
"""
        
        elif analysis_type == 'predictive':
            prompt = f"""You are a data analyst. Summarize the predictive model results:

Target: {data_summary.get('target_column', 'N/A')}
Task Type: {data_summary.get('task_type', 'N/A')}
Best Algorithm: {data_summary.get('best_algorithm', 'N/A')}
Cross-Validation Score: {data_summary.get('cross_validation_score', 'N/A'):.3f}
Top Features: {list(data_summary.get('feature_importance', {}).keys())[:3] if data_summary.get('feature_importance') else 'N/A'}

Summarize the model performance and key drivers:
"""
        
        elif analysis_type == 'prescriptive':
            prompt = f"""You are a data analyst. Provide actionable recommendations:

Target: {data_summary.get('target_column', 'N/A')}
Recommendations: {len(data_summary.get('recommendations', []))}

Provide 2-3 actionable business recommendations:
"""
        
        else:
            prompt = f"""You are a data analyst. Answer this question about the data:

Question: {context}
Data Summary: {str(data_summary)[:500]}

Answer:
"""
        
        return prompt
    
    def answer_question(
        self,
        question: str,
        data_summary: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Answer a question about the data."""
        if self._pipeline is None:
            return {
                'answer': 'LLM not available. Please review the analysis results directly.',
                'confidence': 0.0,
                'suggested_followups': [],
            }
        
        prompt = f"""You are a data analyst assistant. Answer the user's question based on the data analysis.

Dataset Information:
- Rows: {data_summary.get('total_rows', 'N/A')}
- Columns: {data_summary.get('total_columns', 'N/A')}
- Key Metrics: {str(list(data_summary.keys())[:5])}

User Question: {question}

Provide a clear, concise answer:
"""
        
        try:
            response = self._pipeline(prompt)[0]['generated_text']
            generated = response[len(prompt):].strip()
            
            return {
                'answer': generated[:500],
                'confidence': 0.7,
                'suggested_followups': [
                    'What are the key trends?',
                    'Which factors have the most impact?',
                    'What actions should I take?',
                ],
            }
            
        except Exception as e:
            return {
                'answer': f'Error: {str(e)}',
                'confidence': 0.0,
                'suggested_followups': [],
            }
