# Performance Optimization Guide

This guide provides comprehensive performance optimization options for the Climate-Health ML analysis pipeline.

## ⚡ Performance Modes Overview

The analysis pipeline offers several performance optimization modes for different use cases:

### 🚀 Ultra-Fast Mode (~15 seconds)
Perfect for development, testing, and quick iterations:
```bash
python run_analysis.py --fast --skip-deep-learning --synthetic
```
- ✅ Only 3 tree-based models (Random Forest, XGBoost, LightGBM)
- ✅ Minimal model complexity (10 estimators each)
- ✅ 2-fold cross-validation
- ✅ Full evaluation pipeline maintained

### ⚡ Quick Mode (~1-2 minutes)
Good balance of speed and model accuracy:
```bash
python run_analysis.py --quick --synthetic
```
- ✅ Reduced model complexity
- ✅ 3-fold cross-validation  
- ✅ 5-10 epochs for deep learning
- ✅ All 8 models included

### 🏃 Skip Deep Learning (~2-3 minutes)
Full analysis without slow LSTM/GRU models:
```bash
python run_analysis.py --skip-deep-learning --synthetic
```
- ✅ All models except LSTM/GRU
- ✅ Full model complexity
- ✅ Standard cross-validation
- ✅ Complete feature importance analysis

### 🔄 Parallel Processing
Use all CPU cores for faster training:
```bash
python run_analysis.py --parallel --synthetic
```
- ✅ Multi-core model training
- ✅ Significant speedup for tree-based models
- ✅ Combine with other modes

### 💾 Cached Mode  
Cache processed data for faster subsequent runs:
```bash
python run_analysis.py --cache --synthetic
```
- ✅ First run: normal speed + data caching
- ✅ Subsequent runs: ~30% faster

### 🎯 Custom Model Selection
Train only specific models:
```bash
python run_analysis.py --models random_forest xgboost lightgbm --synthetic
```

## 🚀 Recommended Combinations

### For Development:
```bash
python run_analysis.py --fast --parallel --cache --synthetic
```

### For Production Analysis:  
```bash
python run_analysis.py --quick --parallel --synthetic
```

### For Full Analysis (No Deep Learning):
```bash
python run_analysis.py --skip-deep-learning --parallel --synthetic
```

### For Complete Analysis:
```bash
python run_analysis.py --parallel --cache --synthetic
```

## 📊 Performance Comparison

| Mode | Time | Models | Cross-Validation | Use Case |
|------|------|--------|------------------|----------|
| Ultra-Fast | ~15s | 3 tree-based | 2-fold | Development, testing |
| Quick | ~1-2min | All 8 models | 3-fold | Balanced analysis |
| Skip Deep Learning | ~2-3min | 6 models (no LSTM/GRU) | Standard | Full analysis, faster |
| Full | ~5-10min | All 8 models | 5-fold | Complete analysis |

## 💡 Optimization Tips

### Data Size Optimization
- Use `--synthetic` for consistent testing
- Consider data sampling for very large datasets
- Enable `--cache` for repeated runs

### Model-Specific Optimizations
- Skip deep learning models for faster iterations
- Use parallel processing on multi-core systems
- Reduce cross-validation folds for development

### Memory Management
- Monitor memory usage with large datasets
- Use incremental model training for memory-constrained systems
- Clear model cache between runs if needed

## 🔧 Advanced Configuration

### Custom Performance Settings
Edit `config.yaml` to customize:
```yaml
performance:
  n_jobs: -1  # Use all CPU cores
  cv_folds: 3  # Reduce for speed
  max_iterations: 100  # Limit for development
  early_stopping: true  # Stop training early if no improvement
```

### Environment Variables
```bash
export PYTHONUNBUFFERED=1  # Better logging output
export OMP_NUM_THREADS=4   # Control OpenMP threads
export MKL_NUM_THREADS=4   # Control Intel MKL threads
```

## 🚨 Troubleshooting

### Common Issues
- **Out of Memory**: Use `--quick` mode or reduce data size
- **Slow Performance**: Enable `--parallel` and `--cache`
- **Model Training Failures**: Use `--skip-deep-learning` to isolate issues

### Performance Monitoring
```bash
# Monitor system resources during analysis
python run_analysis.py --quick --synthetic & htop
```

All performance modes maintain the complete evaluation pipeline including time series cross-validation, spatial generalization testing, and comprehensive reporting.