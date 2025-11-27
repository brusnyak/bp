#!/usr/bin/env python3
"""
Coqui TTS Chart Generation Script
Generates comprehensive visualizations for thesis documentation.
"""

import json
import os
import re
import matplotlib.pyplot as plt
import numpy as np
from pathlib import Path

# Set style for professional-looking charts
plt.style.use('ggplot')

def parse_optimization_results(filepath: str) -> dict:
    """Parse optimization_results.txt for performance metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {}
    
    # Extract EN→EN metrics
    en_en_first = re.search(r'EN→EN First Call:\s+([0-9.]+)s.*RTF:\s+([0-9.]+)', content)
    en_en_cached = re.search(r'EN→EN Cached:\s+([0-9.]+)s.*RTF:\s+([0-9.]+)', content)
    cache_speedup = re.search(r'Cache Speedup:\s+([0-9.]+)%', content)
    en_sk_cached = re.search(r'EN→SK Cached:\s+([0-9.]+)s.*RTF:\s+([0-9.]+)', content)
    
    if en_en_first:
        data['en_en_first_latency'] = float(en_en_first.group(1))
        data['en_en_first_rtf'] = float(en_en_first.group(2))
    if en_en_cached:
        data['en_en_cached_latency'] = float(en_en_cached.group(1))
        data['en_en_cached_rtf'] = float(en_en_cached.group(2))
    if cache_speedup:
        data['cache_speedup_percent'] = float(cache_speedup.group(1))
    if en_sk_cached:
        data['en_sk_cached_latency'] = float(en_sk_cached.group(1))
        data['en_sk_cached_rtf'] = float(en_sk_cached.group(2))
    
    return data

def parse_tuning_results(filepath: str) -> list:
    """Parse tuning_log.txt for thread/speed benchmark data."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    results = []
    pattern = r'(\d+)\s+([0-9.]+)\s+([0-9.]+)\s+([0-9.]+)'
    
    for match in re.finditer(pattern, content):
        results.append({
            'threads': int(match.group(1)),
            'speed': float(match.group(2)),
            'first_chunk': float(match.group(3)),
            'total_time': float(match.group(4))
        })
    
    return results

def parse_quality_results(filepath: str) -> dict:
    """Parse quality_test_log_final.txt for final quality metrics."""
    with open(filepath, 'r') as f:
        content = f.read()
    
    data = {}
    
    # Extract EN and CS synthesis times
    en_match = re.search(r'Generated cloned_en\.wav in ([0-9.]+)s.*audio duration: ([0-9.]+)s', content)
    cs_match = re.search(r'Generated cloned_cs\.wav in ([0-9.]+)s.*audio duration: ([0-9.]+)s', content)
    
    if en_match:
        data['en_synthesis_time'] = float(en_match.group(1))
        data['en_audio_duration'] = float(en_match.group(2))
        data['en_rtf'] = data['en_synthesis_time'] / data['en_audio_duration']
    
    if cs_match:
        data['cs_synthesis_time'] = float(cs_match.group(1))
        data['cs_audio_duration'] = float(cs_match.group(2))
        data['cs_rtf'] = data['cs_synthesis_time'] / data['cs_audio_duration']
    
    return data

def get_piper_baseline() -> dict:
    """Return Piper TTS baseline metrics from report.txt."""
    return {
        'stt_latency': 0.02,
        'mt_latency': 0.23,
        'tts_latency_sk': 0.30,
        'tts_latency_cs': 0.21,
        'total_e2e': 0.46
    }

def chart1_latency_comparison(coqui_data: dict, piper_data: dict, output_dir: str):
    """Chart 1: Coqui vs Piper - Component Latency Comparison."""
    
    # Assume STT and MT are similar for both (from report.txt)
    stt_latency = 0.02
    mt_latency = 0.23
    
    # Coqui TTS latency (use CS as it's the target language)
    coqui_tts = coqui_data.get('cs_synthesis_time', 8.34)
    coqui_total = stt_latency + mt_latency + coqui_tts
    
    # Piper TTS latency
    piper_tts = piper_data['tts_latency_cs']
    piper_total = piper_data['total_e2e']
    
    labels = ['STT', 'MT', 'TTS', 'Total E2E']
    coqui_latencies = [stt_latency, mt_latency, coqui_tts, coqui_total]
    piper_latencies = [stt_latency, mt_latency, piper_tts, piper_total]
    
    x = np.arange(len(labels))
    width = 0.35
    
    fig, ax = plt.subplots(figsize=(12, 7))
    bars1 = ax.bar(x - width/2, coqui_latencies, width, label='Coqui TTS (Voice Cloning)', color='#3498db')
    bars2 = ax.bar(x + width/2, piper_latencies, width, label='Piper TTS (Generic Voice)', color='#e74c3c')
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Component Latency Comparison: Coqui TTS vs Piper TTS', fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(labels, fontsize=11)
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels on bars
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.2f}s',
                   ha='center', va='bottom', fontsize=9)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart1_latency_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Generated chart1_latency_comparison.png")

def chart2_tuning_heatmap(tuning_data: list, output_dir: str):
    """Chart 2: Thread & Speed Tuning Heatmap."""
    
    # Create pivot table for heatmap
    threads = sorted(set(d['threads'] for d in tuning_data))
    speeds = sorted(set(d['speed'] for d in tuning_data))
    
    # Create matrix for first chunk latency
    matrix = np.zeros((len(speeds), len(threads)))
    
    for d in tuning_data:
        i = speeds.index(d['speed'])
        j = threads.index(d['threads'])
        matrix[i, j] = d['first_chunk']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    im = ax.imshow(matrix, cmap='RdYlGn_r', aspect='auto')
    
    ax.set_xticks(np.arange(len(threads)))
    ax.set_yticks(np.arange(len(speeds)))
    ax.set_xticklabels(threads)
    ax.set_yticklabels([f'{s}x' for s in speeds])
    
    ax.set_xlabel('Thread Count', fontsize=12)
    ax.set_ylabel('Speed Multiplier', fontsize=12)
    ax.set_title('Coqui TTS: First Chunk Latency Tuning (lower is better)', fontsize=14, fontweight='bold')
    
    # Add text annotations
    for i in range(len(speeds)):
        for j in range(len(threads)):
            text = ax.text(j, i, f'{matrix[i, j]:.2f}s',
                          ha="center", va="center", color="white", fontsize=20, fontweight='bold')
    
    # Add colorbar
    cbar = plt.colorbar(im, ax=ax)
    cbar.set_label('Latency (seconds)', fontsize=11)
    
    # Highlight best configuration
    best_idx = np.unravel_index(np.argmin(matrix), matrix.shape)
    rect = plt.Rectangle((best_idx[1]-0.5, best_idx[0]-0.5), 1, 1, 
                         fill=False, edgecolor='blue', linewidth=3)
    ax.add_patch(rect)
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart2_tuning_heatmap.png'), dpi=300)
    plt.close()
    print(f"✅ Generated chart2_tuning_heatmap.png")

def chart3_optimization_impact(opt_data: dict, output_dir: str):
    """Chart 3: Optimization Impact (Before/After)."""
    
    # Baseline (no optimizations) - estimated from first call
    baseline_latency = opt_data.get('en_en_first_latency', 8.31)
    
    # Optimized (with caching)
    optimized_latency = opt_data.get('en_en_cached_latency', 8.15)
    
    # Calculate improvement
    improvement = baseline_latency - optimized_latency
    improvement_pct = (improvement / baseline_latency) * 100
    
    labels = ['Baseline\n(No Cache)', 'Optimized\n(With Cache)']
    latencies = [baseline_latency, optimized_latency]
    colors = ['#e74c3c', '#27ae60']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, latencies, color=colors, width=0.5)
    
    ax.set_ylabel('Latency (seconds)', fontsize=12)
    ax.set_title('Coqui TTS: Speaker Embedding Cache Impact', fontsize=14, fontweight='bold')
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}s',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    # Add improvement annotation
    ax.annotate(f'Improvement: {improvement:.2f}s ({improvement_pct:.1f}%)',
                xy=(0.5, max(latencies) * 0.5),
                ha='center', fontsize=12, 
                bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart3_optimization_impact.png'), dpi=300)
    plt.close()
    print(f"✅ Generated chart3_optimization_impact.png")

def chart4_rtf_comparison(opt_data: dict, quality_data: dict, output_dir: str):
    """Chart 4: Real-Time Factor (RTF) Comparison."""
    
    # Coqui RTF values
    coqui_en_rtf = quality_data.get('en_rtf', opt_data.get('en_en_cached_rtf', 1.72))
    coqui_cs_rtf = quality_data.get('cs_rtf', opt_data.get('en_sk_cached_rtf', 1.72))
    
    # Piper RTF (estimated: 0.21s TTS / ~4.5s audio ≈ 0.047)
    piper_rtf = 0.05
    
    labels = ['Coqui TTS\n(English)', 'Coqui TTS\n(Czech)', 'Piper TTS\n(Generic)']
    rtf_values = [coqui_en_rtf, coqui_cs_rtf, piper_rtf]
    colors = ['#3498db', '#9b59b6', '#e74c3c']
    
    fig, ax = plt.subplots(figsize=(10, 6))
    bars = ax.bar(labels, rtf_values, color=colors, width=0.5)
    
    # Add RTF=1.0 reference line
    ax.axhline(y=1.0, color='green', linestyle='--', linewidth=2, label='Real-time (RTF=1.0)')
    
    ax.set_ylabel('Real-Time Factor (RTF)', fontsize=12)
    ax.set_title('Real-Time Factor Comparison (lower is better)', fontsize=14, fontweight='bold')
    ax.legend(fontsize=11)
    ax.grid(axis='y', alpha=0.3)
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        ax.text(bar.get_x() + bar.get_width()/2., height,
               f'{height:.2f}',
               ha='center', va='bottom', fontsize=11, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart4_rtf_comparison.png'), dpi=300)
    plt.close()
    print(f"✅ Generated chart4_rtf_comparison.png")

def chart5_artifact_timeline(output_dir: str):
    """Chart 5: Artifact Elimination Timeline."""
    
    iterations = [
        'Baseline',
        'Iter 1\n(Params)',
        'Iter 2\n(Revert)',
        'Iter 3\n(Fade)',
        'Iter 4\n(Tuning)',
        'Hybrid\n(Single-shot)',
        'Final\n(Smart Trim)'
    ]
    
    # Subjective artifact severity (0-10, 10 = worst)
    artifact_severity = [10, 7, 8, 5, 6, 3, 0]
    
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(iterations, artifact_severity, marker='o', linewidth=2, markersize=10, color='#e74c3c')
    ax.fill_between(range(len(iterations)), artifact_severity, alpha=0.3, color='#e74c3c')
    
    ax.set_ylabel('Artifact Severity (0=clean, 10=worst)', fontsize=12)
    ax.set_xlabel('Iteration', fontsize=12)
    ax.set_title('Artifact Elimination Progress', fontsize=14, fontweight='bold')
    ax.grid(True, alpha=0.3)
    
    # Add annotations for key milestones
    annotations = [
        (0, 10, '"kachunk" noise'),
        (3, 5, 'Fade-out applied'),
        (5, 3, 'Single-shot synthesis'),
        (6, 0, 'Clean audio ✓')
    ]
    
    for x, y, text in annotations:
        ax.annotate(text, xy=(x, y), xytext=(x, y+1),
                   ha='center', fontsize=9,
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7),
                   arrowprops=dict(arrowstyle='->', color='black'))
    
    plt.tight_layout()
    plt.savefig(os.path.join(output_dir, 'chart5_artifact_timeline.png'), dpi=300)
    plt.close()
    print(f"✅ Generated chart5_artifact_timeline.png")

def main():
    """Main execution function."""
    
    # Setup paths
    base_dir = Path(__file__).parent.parent
    test_output_dir = base_dir / 'test_output'
    output_dir = base_dir / 'documentation' / 'visuals'
    output_dir.mkdir(parents=True, exist_ok=True)
    
    print("="*80)
    print("COQUI TTS CHART GENERATION")
    print("="*80)
    
    # Parse data
    print("\n📊 Parsing test data...")
    opt_data = parse_optimization_results(test_output_dir / 'optimization_results.txt')
    tuning_data = parse_tuning_results(test_output_dir / 'tuning_log.txt')
    quality_data = parse_quality_results(test_output_dir / 'quality_test_log_final.txt')
    piper_data = get_piper_baseline()
    
    print(f"  ✓ Optimization data: {len(opt_data)} metrics")
    print(f"  ✓ Tuning data: {len(tuning_data)} configurations")
    print(f"  ✓ Quality data: {len(quality_data)} metrics")
    
    # Generate charts
    print("\n🎨 Generating charts...")
    chart1_latency_comparison(quality_data, piper_data, output_dir)
    chart2_tuning_heatmap(tuning_data, output_dir)
    chart3_optimization_impact(opt_data, output_dir)
    chart4_rtf_comparison(opt_data, quality_data, output_dir)
    chart5_artifact_timeline(output_dir)
    
    print("\n" + "="*80)
    print(f"✅ All charts generated successfully in: {output_dir}")
    print("="*80)

if __name__ == "__main__":
    main()
