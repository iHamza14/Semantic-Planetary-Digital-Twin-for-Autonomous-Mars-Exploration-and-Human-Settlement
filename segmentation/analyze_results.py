import json
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
import argparse

def load_metrics(metrics_path='runs/metrics.json'):
    """Load training metrics from JSON"""
    with open(metrics_path, 'r') as f:
        metrics = json.load(f)
    return metrics

def create_detailed_plots(metrics):
    """Create comprehensive analysis plots"""
    fig = plt.figure(figsize=(18, 10))
    
    epochs = range(1, len(metrics['train_losses']) + 1)
    
    # 1. Training and Validation Loss
    ax1 = plt.subplot(2, 3, 1)
    ax1.plot(epochs, metrics['train_losses'], 'b-', label='Training Loss', linewidth=2)
    ax1.plot(epochs, metrics['val_losses'], 'r-', label='Validation Loss', linewidth=2)
    ax1.set_xlabel('Epoch', fontsize=12)
    ax1.set_ylabel('Loss', fontsize=12)
    ax1.set_title('Training vs Validation Loss', fontsize=14, fontweight='bold')
    ax1.legend(fontsize=10)
    ax1.grid(True, alpha=0.3)
    
    # 2. Validation IoU over time
    ax2 = plt.subplot(2, 3, 2)
    ax2.plot(epochs, metrics['val_ious'], 'g-', linewidth=2, marker='o', markersize=4)
    ax2.axhline(y=metrics['best_iou'], color='r', linestyle='--', 
                label=f'Best IoU: {metrics["best_iou"]:.4f}')
    ax2.set_xlabel('Epoch', fontsize=12)
    ax2.set_ylabel('IoU Score', fontsize=12)
    ax2.set_title('Validation IoU Progress', fontsize=14, fontweight='bold')
    ax2.legend(fontsize=10)
    ax2.grid(True, alpha=0.3)
    
    # 3. Loss difference (overfitting indicator)
    ax3 = plt.subplot(2, 3, 3)
    loss_diff = np.array(metrics['val_losses']) - np.array(metrics['train_losses'])
    ax3.plot(epochs, loss_diff, 'purple', linewidth=2)
    ax3.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax3.set_xlabel('Epoch', fontsize=12)
    ax3.set_ylabel('Val Loss - Train Loss', fontsize=12)
    ax3.set_title('Overfitting Indicator', fontsize=14, fontweight='bold')
    ax3.grid(True, alpha=0.3)
    
    # 4. Learning rate of improvement
    ax4 = plt.subplot(2, 3, 4)
    iou_improvements = [0] + [metrics['val_ious'][i] - metrics['val_ious'][i-1] 
                              for i in range(1, len(metrics['val_ious']))]
    ax4.bar(epochs, iou_improvements, color='teal', alpha=0.7)
    ax4.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
    ax4.set_xlabel('Epoch', fontsize=12)
    ax4.set_ylabel('IoU Change', fontsize=12)
    ax4.set_title('IoU Improvement per Epoch', fontsize=14, fontweight='bold')
    ax4.grid(True, alpha=0.3)
    
    # 5. Summary statistics
    ax5 = plt.subplot(2, 3, 5)
    ax5.axis('off')
    
    stats_text = f"""
    TRAINING SUMMARY
    ───────────────────────────
    
    Total Epochs: {len(metrics['train_losses'])}
    
    Best Validation IoU: {metrics['best_iou']:.4f}
    Final Validation IoU: {metrics['val_ious'][-1]:.4f}
    
    Final Train Loss: {metrics['train_losses'][-1]:.4f}
    Final Val Loss: {metrics['val_losses'][-1]:.4f}
    
    Best Epoch: {np.argmax(metrics['val_ious']) + 1}
    
    Avg IoU Improvement: {np.mean(iou_improvements):.6f}
    
    Loss Difference: {loss_diff[-1]:.4f}
    """
    
    ax5.text(0.1, 0.5, stats_text, fontsize=11, family='monospace',
             verticalalignment='center')
    
    # 6. Performance indicators
    ax6 = plt.subplot(2, 3, 6)
    
    categories = ['Best IoU', 'Final IoU', 'Avg IoU']
    values = [
        metrics['best_iou'],
        metrics['val_ious'][-1],
        np.mean(metrics['val_ious'])
    ]
    colors = ['gold', 'silver', 'lightblue']
    
    bars = ax6.barh(categories, values, color=colors, edgecolor='black', linewidth=1.5)
    ax6.set_xlabel('IoU Score', fontsize=12)
    ax6.set_title('Performance Metrics', fontsize=14, fontweight='bold')
    ax6.set_xlim(0, 1.0)
    
    # Add value labels on bars
    for bar, value in zip(bars, values):
        ax6.text(value + 0.02, bar.get_y() + bar.get_height()/2, 
                f'{value:.4f}', va='center', fontsize=10, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('runs/detailed_analysis.png', dpi=200, bbox_inches='tight')
    print('Detailed analysis saved to runs/detailed_analysis.png')
    plt.show()

def generate_report(metrics):
    """Generate a text report"""
    report = f"""
╔══════════════════════════════════════════════════════════════╗
║          DUALITY AI SEGMENTATION - PERFORMANCE REPORT        ║
╚══════════════════════════════════════════════════════════════╝

TRAINING CONFIGURATION
────────────────────────────────────────────────────────────────
Total Epochs Trained: {len(metrics['train_losses'])}
Model Architecture: U-Net

FINAL PERFORMANCE METRICS
────────────────────────────────────────────────────────────────
Best Validation IoU:     {metrics['best_iou']:.4f}
Final Validation IoU:    {metrics['val_ious'][-1]:.4f}
Average Validation IoU:  {np.mean(metrics['val_ious']):.4f}

Final Training Loss:     {metrics['train_losses'][-1]:.4f}
Final Validation Loss:   {metrics['val_losses'][-1]:.4f}

TRAINING DYNAMICS
────────────────────────────────────────────────────────────────
Best Epoch Number:       {np.argmax(metrics['val_ious']) + 1}
Total IoU Improvement:   {metrics['val_ious'][-1] - metrics['val_ious'][0]:.4f}
Peak IoU Achieved:       {max(metrics['val_ious']):.4f}

MODEL ASSESSMENT
────────────────────────────────────────────────────────────────
"""
    
    # Determine model quality
    best_iou = metrics['best_iou']
    if best_iou >= 0.70:
        assessment = "EXCELLENT - Ready for submission!"
    elif best_iou >= 0.60:
        assessment = "GOOD - Competitive performance"
    elif best_iou >= 0.50:
        assessment = "FAIR - Consider more training"
    else:
        assessment = "NEEDS IMPROVEMENT - Review hyperparameters"
    
    report += f"Overall Assessment:      {assessment}\n"
    report += f"Competition Readiness:   {'✓ YES' if best_iou >= 0.60 else '✗ Needs Work'}\n\n"
    
    # Overfitting check
    loss_diff = metrics['val_losses'][-1] - metrics['train_losses'][-1]
    if loss_diff > 0.3:
        report += "⚠ WARNING: Possible overfitting detected\n"
        report += "  Recommendation: Add regularization or data augmentation\n\n"
    
    # Convergence check
    recent_iou_change = abs(metrics['val_ious'][-1] - metrics['val_ious'][-5])
    if recent_iou_change < 0.01:
        report += "✓ Model has converged\n\n"
    else:
        report += "ℹ Model may benefit from additional training\n\n"
    
    report += "════════════════════════════════════════════════════════════════\n"
    
    return report

def compare_runs(run_dirs):
    """Compare multiple training runs"""
    fig, axes = plt.subplots(1, 2, figsize=(15, 5))
    
    for run_dir in run_dirs:
        metrics_path = Path(run_dir) / 'metrics.json'
        if metrics_path.exists():
            metrics = load_metrics(metrics_path)
            label = Path(run_dir).name
            
            epochs = range(1, len(metrics['train_losses']) + 1)
            axes[0].plot(epochs, metrics['val_losses'], label=label, linewidth=2)
            axes[1].plot(epochs, metrics['val_ious'], label=label, linewidth=2)
    
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Validation Loss')
    axes[0].set_title('Validation Loss Comparison')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Validation IoU')
    axes[1].set_title('Validation IoU Comparison')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.savefig('runs/comparison.png', dpi=150)
    print('Comparison saved to runs/comparison.png')
    plt.show()

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Analyze training results')
    parser.add_argument('--metrics', type=str, default='runs/metrics.json',
                       help='Path to metrics.json file')
    parser.add_argument('--compare', nargs='+', 
                       help='Compare multiple run directories')
    
    args = parser.parse_args()
    
    if args.compare:
        print('Comparing multiple runs...')
        compare_runs(args.compare)
    else:
        if not Path(args.metrics).exists():
            print(f'Error: Metrics file not found at {args.metrics}')
            print('Please train the model first using train.py')
            exit(1)
        
        print('Loading metrics...')
        metrics = load_metrics(args.metrics)
        
        print('\nGenerating detailed analysis...')
        create_detailed_plots(metrics)
        
        print('\n' + '='*65)
        report = generate_report(metrics)
        print(report)
        
        # Save report to file
        with open('runs/performance_report.txt', 'w') as f:
            f.write(report)
        print('Report saved to runs/performance_report.txt')