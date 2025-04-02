import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from scipy import stats

class UtilizationVisualizer:
    def __init__(self, data_a, data_b, name_a="Option A", name_b="Option B", x_tick_labels=None):
        self.data_a = data_a
        self.data_b = data_b
        self.name_a = name_a
        self.name_b = name_b
        
        self.x_tick_labels = x_tick_labels
        
        is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
        
        use_hex_a = False
        samples_a = []
        if data_a['total_utilisation']:
            samples_a.append(data_a['total_utilisation'][0]['TotalUtilisation'])
        for pd_data in data_a['pd_utilisation'].values():
            if pd_data:
                samples_a.append(pd_data[0]['TotalUtilisation'])
                break
        
        for sample in samples_a:
            if is_hex_format(sample):
                use_hex_a = True
                break
        
        use_hex_b = False
        samples_b = []
        if data_b['total_utilisation']:
            samples_b.append(data_b['total_utilisation'][0]['TotalUtilisation'])
        for pd_data in data_b['pd_utilisation'].values():
            if pd_data:
                samples_b.append(pd_data[0]['TotalUtilisation'])
                break
        
        for sample in samples_b:
            if is_hex_format(sample):
                use_hex_b = True
                break
        
        base_a = 16 if use_hex_a else 10
        base_b = 16 if use_hex_b else 10

        self.total_util_a = [int(data['TotalUtilisation'], base_a) for data in data_a['total_utilisation']]
        self.total_util_b = [int(data['TotalUtilisation'], base_b) for data in data_b['total_utilisation']]
        
        self.pd_data_a = {pd_name: [int(m['TotalUtilisation'], base_a) for m in measurements]
                        for pd_name, measurements in data_a['pd_utilisation'].items()}
        
        self.pd_data_b = {pd_name: [int(m['TotalUtilisation'], base_b) for m in measurements]
                        for pd_name, measurements in data_b['pd_utilisation'].items()}
        
        self.pd_percent_a = {}
        self.pd_percent_b = {}
        
        for pd_name, measurements in self.pd_data_a.items():
            self.pd_percent_a[pd_name] = np.array(measurements) / np.array(self.total_util_a) * 100
        
        for pd_name, measurements in self.pd_data_b.items():
            self.pd_percent_b[pd_name] = np.array(measurements) / np.array(self.total_util_b) * 100
        
        self.all_pds = sorted(set(self.pd_data_a.keys()), reverse=True)
        self.highlighted_pds = []
        
        self.pd_colors = {}
        colors = plt.cm.tab10.colors
        for i, pd_name in enumerate(self.all_pds):
            self.pd_colors[pd_name] = colors[i % len(colors)]
    
    def set_x_tick_labels(self, x_tick_labels):
        self.x_tick_labels = x_tick_labels
    
    def print_available_pds(self):
        print("Available PD IDs:")
        for pd_name in self.all_pds:
            print(f"  - {pd_name}")
    
    def set_pds_to_visualize(self, pd_ids):
        self.highlighted_pds = pd_ids
        
        for pd_id in pd_ids:
            if pd_id not in self.all_pds:
                print(f"Warning: PD ID '{pd_id}' not found in dataset")
    
    def add_pd_to_visualize(self, pd_id):
        if pd_id in self.all_pds:
            if pd_id not in self.highlighted_pds:
                self.highlighted_pds.append(pd_id)
        else:
            print(f"Warning: PD ID '{pd_id}' not found in dataset")
    
    def remove_pd_to_visualize(self, pd_id):
        if pd_id in self.highlighted_pds:
            self.highlighted_pds.remove(pd_id)
    
    def clear_pds_to_visualize(self):
        self.highlighted_pds = []
    
    def print_pd_percentages(self):
        print(f"{self.name_a} PD Utilization Percentages:")
        for pd_name in self.all_pds:
            if pd_name in self.pd_percent_a:
                print(f"  {pd_name}: {self.pd_percent_a[pd_name]}")
        
        print(f"\n{self.name_b} PD Utilization Percentages:")
        for pd_name in self.all_pds:
            if pd_name in self.pd_percent_b:
                print(f"  {pd_name}: {self.pd_percent_b[pd_name]}")
    
    def _set_x_ticks(self, ax, iterations):
        if self.x_tick_labels is not None:
            if len(self.x_tick_labels) >= len(iterations):
                ax.set_xticks(iterations)
                ax.set_xticklabels(self.x_tick_labels[:len(iterations)])
            else:
                print(f"Warning: Not enough x_tick_labels provided. Expected {len(iterations)}, got {len(self.x_tick_labels)}")
                ax.set_xticks(iterations)
        else:
            ax.set_xticks(iterations)
    
    def plot_all(self, pd_ids=None, save_path=None, figsize=(12, 8), dpi=150, driver_name=None, use_percent=False):
        plt.figure(figsize=figsize, dpi=dpi)
        ax = plt.gca()
        
        pds_to_plot = pd_ids if pd_ids is not None else (self.highlighted_pds if self.highlighted_pds else self.all_pds)
        
        colors = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b', '#e377c2', '#7f7f7f']
        
        pd_handles = []
        pd_labels = []
        
        plotted_pds = []
        
        if use_percent:
            data_a = self.pd_percent_a
            data_b = self.pd_percent_b
        else:
            data_a = self.pd_data_a
            data_b = self.pd_data_b
        
        for i, pd_name in enumerate(pds_to_plot):
            if pd_name not in self.all_pds:
                print(f"Warning: PD ID '{pd_name}' not found in dataset")
                continue
            
            color = colors[i % len(colors)]
            was_plotted = False
            
            if pd_name in self.pd_percent_a:
                iterations = range(1, len(data_a[pd_name]) + 1)
                line_a = ax.plot(iterations, data_a[pd_name], '--o', 
                        color=color, 
                        linewidth=2, 
                        markersize=6)
                was_plotted = True

            if pd_name in self.pd_percent_b:
                iterations = range(1, len(data_b[pd_name]) + 1)
                line_b = ax.plot(iterations, data_b[pd_name], '-s', 
                        color=color, 
                        linewidth=2, 
                        markersize=6)
                was_plotted = True
            
            if was_plotted:
                plotted_pds.append(pd_name)
                pd_line = Line2D([0], [0], color=color, linewidth=2.5)
                pd_handles.append(pd_line)
                pd_labels.append(pd_name)
        
        title_str = f"PD Utilisation{f' ({driver_name})' if driver_name else ''}: {self.name_a} vs {self.name_b}"        
        ax.set_title(title_str, fontsize=14, fontweight='bold')
        ax.set_xlabel('Throughput', fontsize=12)
        ax.set_ylabel('CPU Utilisation (cycles)', fontsize=12)
        ax.grid(True, linestyle='--', alpha=0.7)
        
        iterations = range(1, max([
            len(data) for data in [*self.pd_percent_a.values(), *self.pd_percent_b.values()]
        ], default=1) + 1)
        
        self._set_x_ticks(ax, iterations)
        
        ax.set_ylim(bottom=0)
        
        style_a = Line2D([0], [0], linestyle='--', marker='o', color='black', markersize=6)
        style_b = Line2D([0], [0], linestyle='-', marker='s', color='black', markersize=6)
        
        implementation_legend = ax.legend([style_a, style_b], [self.name_a, self.name_b], 
                                        loc='upper left', fontsize=10)
        ax.add_artist(implementation_legend)
        
        ax.legend(pd_handles, pd_labels, 
                loc='upper center', bbox_to_anchor=(0.5, -0.15),
                ncol=min(len(pd_handles), 5), fontsize=10)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()   

    
    def plot_highlighted(self, save_path=None, figsize=(19, 15), dpi=500, show_max_diff_for=None):
        if not self.highlighted_pds:
            print("No PDs selected for highlighting. Use set_pds_to_visualize() or add_pd_to_visualize() first.")
            return
            
        self.plot_all(pd_ids=self.highlighted_pds, save_path=save_path, figsize=figsize, dpi=dpi, 
                     show_max_diff_for=show_max_diff_for)
        
        if show_max_diff_for and show_max_diff_for in self.pd_percent_a and show_max_diff_for in self.pd_percent_b:
            a_data = self.pd_percent_a[show_max_diff_for]
            b_data = self.pd_percent_b[show_max_diff_for]
            
            min_len = min(len(a_data), len(b_data))
            diffs = [abs(a_data[i] - b_data[i]) for i in range(min_len)]
            avg_diff = np.mean(diffs)
            
            smaller_vals = [min(a_data[i], b_data[i]) for i in range(min_len)]
            avg_smaller = np.mean([v for v in smaller_vals if v > 0])
            avg_diff_rel = (avg_diff / avg_smaller) * 100 if avg_smaller > 0 else 0
            
            print(f"Average difference for {show_max_diff_for}: {avg_diff:.1f}% points (relative: {avg_diff_rel:.1f}%)")

    def plot_specific_pd(self, pd_name, save_path=None, figsize=(12, 8), dpi=100, show_max_diff=False, driver_name=None, use_percent=False):
        if pd_name not in self.pd_percent_a or pd_name not in self.pd_percent_b:
            print(f"Error: PD '{pd_name}' not found in both datasets")
            return
            
        plt.figure(figsize=figsize, dpi=dpi)
        
        ax1 = plt.gca()
        
        iterations = range(1, min(len(self.pd_percent_a[pd_name]), len(self.pd_percent_b[pd_name])) + 1)
        width = 0.35
        x = np.array(iterations)
        
        if use_percent:
            a_data = self.pd_percent_a[pd_name][:len(iterations)]
            b_data = self.pd_percent_b[pd_name][:len(iterations)]
        else:
            a_data = self.pd_data_a[pd_name][:len(iterations)]
            b_data = self.pd_data_b[pd_name][:len(iterations)]
        
        bars1 = ax1.bar(x - width/2, a_data, width, label=self.name_a, color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x + width/2, b_data, width, label=self.name_b, color='darkorange', alpha=0.8)
        
        abs_diffs = []
        rel_diffs = []
        
        for i in range(len(iterations)):
            a_val = a_data[i]
            b_val = b_data[i]
            
            print(f"Iteration {i+1}: {pd_name} - {self.name_a}: {a_val}, {self.name_b}: {b_val}")
            
            abs_diff = a_val - b_val
            abs_diffs.append(abs(abs_diff))
            
            rel_diff = (abs_diff / b_val) * 100 if b_val > 0 else 0
            rel_diffs.append(abs(rel_diff))
            
        print(rel_diffs)
        
        avg_abs_diff = np.mean(abs_diffs)
        
        epsilon = 1e-10
        rel_diffs_no_zeros = [max(x, epsilon) for x in rel_diffs]
        
        avg_rel_diff = stats.gmean(rel_diffs_no_zeros)
        
        a_vs_b = np.mean([a_data[i] - b_data[i] for i in range(len(iterations))])
        avg_text = "higher" if a_vs_b > 0 else "lower"
        
        arrow = "↑" if a_vs_b > 0 else "↓"
        color = "red" if a_vs_b > 0 else "green"
        sign = "+" if a_vs_b > 0 else "-"
        
        if use_percent:
            diff_text = f"Average Δ: {arrow} {avg_abs_diff:.1f}% (Geom. {sign}{avg_rel_diff:.1f}%)"        
        else:
            def format_number(num):
                if abs(num) == 0:
                    return "0"
                return f"{num:.2e}"
                
            diff_text = f"Average Δ: {arrow} {format_number(avg_abs_diff)} (Geom. {sign}{avg_rel_diff:.1f}%)"
        
        props = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=color)
        ax1.text(0.98, 0.95, diff_text, transform=ax1.transAxes, fontsize=10,
               verticalalignment='top', horizontalalignment='right', color=color, bbox=props)
        
        pd_name_title = ' '.join(word.capitalize() for word in pd_name.split('_')).split('(')[0].strip()
        if driver_name:
            pd_name_title += f" ({driver_name})"
        ax1.set_title(f'{pd_name_title} Utilisation: {self.name_a} vs {self.name_b}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Throughput', fontsize=12)
        
        if use_percent:
            ax1.set_ylabel('Percentage of Total Utilisation (%)', fontsize=12)
        else:
            ax1.set_ylabel('CPU Utilisation (cycles)', fontsize=12)
            
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        self._set_x_ticks(ax1, x)
        
        max_height = max(max(a_data), max(b_data))
        y_limit = max_height * 1.5
        ax1.set_ylim(0, y_limit)
        
        for i, (bar1, bar2) in enumerate(zip(bars1, bars2)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            
            
            if abs(height1 - height2) > 0:
                diff = height1 - height2
                diff_pct = (diff / height2) * 100 if height2 > 0 else 0
                
                if diff > 0:
                    direction = "↑"
                    color = "red"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                else:
                    direction = "↓"
                    color = "green"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                
                mid_x = (bar1.get_x() + bar1.get_width() + bar2.get_x()) / 2
                max_y = max(height1, height2)
                box_y = max_y + (y_limit * 0.05)
                
                ax1.annotate(diff_text,
                          xy=(mid_x, box_y),
                          xytext=(0, 0),
                          textcoords="offset points",
                          ha='center', va='bottom',
                          fontsize=8, color=color,
                          bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7))
        
        ax1.legend(loc='upper left', ncol=2, frameon=True)
        
        if show_max_diff:
            max_diff_idx = np.argmax(abs_diffs)
            max_diff = abs_diffs[max_diff_idx]
            max_rel_diff = rel_diffs[max_diff_idx]
            
            if max_diff > 0:
                x_pos = x[max_diff_idx]
                ax1.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.15, color='red')
                
                bar1 = bars1[max_diff_idx]
                bar2 = bars2[max_diff_idx]
                
                highest_annotation = max([
                    max(bar1.get_height(), bar2.get_height()) + (y_limit * 0.1)
                    for bar1, bar2 in zip(bars1, bars2)
                ])
                
                y_pos = highest_annotation + (y_limit * 0.05)
                ax1.annotate(f"Max diff", 
                           xy=(x_pos, y_pos),
                           xytext=(0, 0),
                           textcoords="offset points",
                           ha='center', va='bottom',
                           fontsize=9, color='red',
                           bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="red", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()