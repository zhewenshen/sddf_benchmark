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
        
        
    def plot_specific_pd_with_new_data(self, pd_name, new_data, new_data_name, save_path=None, figsize=(12, 8), dpi=100, show_max_diff=False, driver_name=None, use_percent=False):
        if pd_name not in self.pd_percent_a or pd_name not in self.pd_percent_b or pd_name not in new_data['pd_utilisation']:
            print(f"Error: PD '{pd_name}' not found in all datasets")
            return
            
        plt.figure(figsize=figsize, dpi=dpi)
        
        ax1 = plt.gca()
        
        min_len = min(len(self.pd_percent_a[pd_name]), len(self.pd_percent_b[pd_name]), len(new_data['pd_utilisation'][pd_name]))
        iterations = range(1, min_len + 1)
        width = 0.25
        x = np.array(iterations)
        
        if use_percent:
            a_data = self.pd_percent_a[pd_name][:min_len]
            b_data = self.pd_percent_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            total_util_new = [int(data['TotalUtilisation'], base_new) for data in new_data['total_utilisation']]
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            pd_percent_new = {}
            for pd, measurements in pd_data_new.items():
                pd_percent_new[pd] = np.array(measurements) / np.array(total_util_new) * 100
                
            new_data_values = pd_percent_new[pd_name][:min_len]
        else:
            a_data = self.pd_data_a[pd_name][:min_len]
            b_data = self.pd_data_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            new_data_values = pd_data_new[pd_name][:min_len]
        
        bars1 = ax1.bar(x - width, a_data, width, label=self.name_a, color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x, b_data, width, label=self.name_b, color='darkorange', alpha=0.8)
        bars3 = ax1.bar(x + width, new_data_values, width, label=new_data_name, color='forestgreen', alpha=0.8)
        
        # Calculate differences relative to baseline (data_b)
        a_diffs = []
        new_diffs = []
        
        for i in range(len(iterations)):
            a_val = a_data[i]
            b_val = b_data[i]
            new_val = new_data_values[i]
            
            print(f"Iteration {i+1}: {pd_name} - {self.name_a}: {a_val}, {self.name_b}: {b_val}, {new_data_name}: {new_val}")
            
            a_diff = a_val - b_val
            a_diffs.append(abs(a_diff))
            
            new_diff = new_val - b_val
            new_diffs.append(abs(new_diff))
        
        # Calculate average differences
        avg_a_diff = np.mean([a_data[i] - b_data[i] for i in range(len(iterations))])
        avg_new_diff = np.mean([new_data_values[i] - b_data[i] for i in range(len(iterations))])
        
        # Create difference indicators
        max_height = max(max(a_data), max(b_data), max(new_data_values))
        y_limit = max_height * 1.5
        ax1.set_ylim(0, y_limit)
        
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            height3 = bar3.get_height()
            
            # Difference between A and B
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
            
            # Difference between C and B
            if abs(height3 - height2) > 0:
                diff = height3 - height2
                diff_pct = (diff / height2) * 100 if height2 > 0 else 0
                
                if diff > 0:
                    direction = "↑"
                    color = "red"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                else:
                    direction = "↓"
                    color = "green"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                
                mid_x = (bar3.get_x() + bar3.get_width() + bar2.get_x() + bar2.get_width()) / 2
                max_y = max(height3, height2)
                box_y = max_y + (y_limit * 0.05)
                
                ax1.annotate(diff_text,
                        xy=(mid_x, box_y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7))
        
        # Add overall difference indicators
        a_arrow = "↑" if avg_a_diff > 0 else "↓"
        a_color = "red" if avg_a_diff > 0 else "green"
        a_sign = "+" if avg_a_diff > 0 else "-"
        
        new_arrow = "↑" if avg_new_diff > 0 else "↓"
        new_color = "red" if avg_new_diff > 0 else "green"
        new_sign = "+" if avg_new_diff > 0 else "-"
        
        epsilon = 1e-10
        a_rel_diffs_no_zeros = [max(abs((a_data[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        new_rel_diffs_no_zeros = [max(abs((new_data_values[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        
        avg_a_rel_diff = stats.gmean(a_rel_diffs_no_zeros)
        avg_new_rel_diff = stats.gmean(new_rel_diffs_no_zeros)
        
        if use_percent:
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {abs(avg_a_diff):.1f}% (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {abs(avg_new_diff):.1f}% (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        else:
            def format_number(num):
                if abs(num) == 0:
                    return "0"
                return f"{num:.2e}"
                
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {format_number(abs(avg_a_diff))} (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {format_number(abs(avg_new_diff))} (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        
        props_a = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=a_color)
        props_new = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=new_color)
        
        ax1.text(0.98, 0.95, a_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=a_color, bbox=props_a)
        ax1.text(0.98, 0.88, new_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=new_color, bbox=props_new)
        
        pd_name_title = ' '.join(word.capitalize() for word in pd_name.split('_')).split('(')[0].strip()
        if driver_name:
            pd_name_title += f" ({driver_name})"
        ax1.set_title(f'{pd_name_title} Utilisation: {self.name_a} vs {self.name_b} vs {new_data_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Throughput', fontsize=12)
        
        if use_percent:
            ax1.set_ylabel('Percentage of Total Utilisation (%)', fontsize=12)
        else:
            ax1.set_ylabel('CPU Utilisation (cycles)', fontsize=12)
            
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        self._set_x_ticks(ax1, x)
        
        ax1.legend(loc='upper left', ncol=3, frameon=True)
        
        # Show max diff if requested
        if show_max_diff:
            a_max_diff_idx = np.argmax([abs(a_data[i] - b_data[i]) for i in range(len(iterations))])
            new_max_diff_idx = np.argmax([abs(new_data_values[i] - b_data[i]) for i in range(len(iterations))])
            
            if a_max_diff_idx == new_max_diff_idx:
                x_pos = x[a_max_diff_idx]
                ax1.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.15, color='purple')
                ax1.annotate(f"Max diff (both)", 
                        xy=(x_pos, y_limit * 0.95),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='purple',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="purple", alpha=0.8))
            else:
                # A max diff
                x_pos_a = x[a_max_diff_idx]
                ax1.axvspan(x_pos_a-0.5, x_pos_a+0.5, alpha=0.15, color='steelblue')
                ax1.annotate(f"Max diff ({self.name_a})", 
                        xy=(x_pos_a, y_limit * 0.95),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='steelblue',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="steelblue", alpha=0.8))
                
                # New max diff
                x_pos_new = x[new_max_diff_idx]
                ax1.axvspan(x_pos_new-0.5, x_pos_new+0.5, alpha=0.15, color='forestgreen')
                ax1.annotate(f"Max diff ({new_data_name})", 
                        xy=(x_pos_new, y_limit * 0.88),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='forestgreen',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="forestgreen", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        
    def plot_specific_pd_with_new_data_2(self, pd_name, new_data, new_data_name, save_path=None, figsize=(12, 8), dpi=100, show_max_diff=False, driver_name=None, use_percent=False):
        if pd_name not in self.pd_percent_a or pd_name not in self.pd_percent_b or pd_name not in new_data['pd_utilisation']:
            print(f"Error: PD '{pd_name}' not found in all datasets")
            return
            
        plt.figure(figsize=figsize, dpi=dpi)
        
        ax1 = plt.gca()
        
        min_len = min(len(self.pd_percent_a[pd_name]), len(self.pd_percent_b[pd_name]), len(new_data['pd_utilisation'][pd_name]))
        iterations = range(1, min_len + 1)
        width = 0.25
        x = np.array(iterations)
        
        if use_percent:
            a_data = self.pd_percent_a[pd_name][:min_len]
            b_data = self.pd_percent_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            total_util_new = [int(data['TotalUtilisation'], base_new) for data in new_data['total_utilisation']]
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            pd_percent_new = {}
            for pd, measurements in pd_data_new.items():
                pd_percent_new[pd] = np.array(measurements) / np.array(total_util_new) * 100
                
            new_data_values = pd_percent_new[pd_name][:min_len]
        else:
            a_data = self.pd_data_a[pd_name][:min_len]
            b_data = self.pd_data_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            new_data_values = pd_data_new[pd_name][:min_len]
        
        bars1 = ax1.bar(x - width, a_data, width, label=self.name_a, color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x, b_data, width, label=self.name_b, color='darkorange', alpha=0.8)
        bars3 = ax1.bar(x + width, new_data_values, width, label=new_data_name, color='forestgreen', alpha=0.8)
        
        # Calculate differences relative to baseline (data_b)
        a_diffs = []
        new_diffs = []
        
        for i in range(len(iterations)):
            a_val = a_data[i]
            b_val = b_data[i]
            new_val = new_data_values[i]
            
            print(f"Iteration {i+1}: {pd_name} - {self.name_a}: {a_val}, {self.name_b}: {b_val}, {new_data_name}: {new_val}")
            
            a_diff = a_val - b_val
            a_diffs.append(abs(a_diff))
            
            new_diff = new_val - b_val
            new_diffs.append(abs(new_diff))
        
        # Calculate average differences
        avg_a_diff = np.mean([a_data[i] - b_data[i] for i in range(len(iterations))])
        avg_new_diff = np.mean([new_data_values[i] - b_data[i] for i in range(len(iterations))])
        
        # Create difference indicators with staggered positioning
        max_height = max(max(a_data), max(b_data), max(new_data_values))
        y_limit = max_height * 1.5
        ax1.set_ylim(0, y_limit)
        
        # Use staggered heights for annotations to prevent overlap
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            height3 = bar3.get_height()
            
            # Difference between A and B - positioned lower
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
                
                # Position annotation over bar 1 instead of between bars for better spacing
                x_pos = bar1.get_x() + bar1.get_width()/2
                max_y = max(height1, height2)
                box_y = max_y + (y_limit * 0.02)  # Lower position
                
                ax1.annotate(diff_text,
                        xy=(x_pos, box_y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7))
            
            # Difference between C and B - positioned higher
            if abs(height3 - height2) > 0:
                diff = height3 - height2
                diff_pct = (diff / height2) * 100 if height2 > 0 else 0
                
                if diff > 0:
                    direction = "↑"
                    color = "red"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                else:
                    direction = "↓"
                    color = "green"
                    diff_text = f"{direction} {abs(diff_pct):.1f}%"
                
                # Position annotation over bar 3 instead of between bars for better spacing
                x_pos = bar3.get_x() + bar3.get_width()/2
                max_y = max(height3, height2)
                box_y = max_y + (y_limit * 0.08)  # Higher position
                
                ax1.annotate(diff_text,
                        xy=(x_pos, box_y),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=8, color=color,
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec=color, alpha=0.7))
        
        # Add overall difference indicators
        a_arrow = "↑" if avg_a_diff > 0 else "↓"
        a_color = "red" if avg_a_diff > 0 else "green"
        a_sign = "+" if avg_a_diff > 0 else "-"
        
        new_arrow = "↑" if avg_new_diff > 0 else "↓"
        new_color = "red" if avg_new_diff > 0 else "green"
        new_sign = "+" if avg_new_diff > 0 else "-"
        
        epsilon = 1e-10
        a_rel_diffs_no_zeros = [max(abs((a_data[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        new_rel_diffs_no_zeros = [max(abs((new_data_values[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        
        avg_a_rel_diff = stats.gmean(a_rel_diffs_no_zeros)
        avg_new_rel_diff = stats.gmean(new_rel_diffs_no_zeros)
        
        if use_percent:
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {abs(avg_a_diff):.1f}% (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {abs(avg_new_diff):.1f}% (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        else:
            def format_number(num):
                if abs(num) == 0:
                    return "0"
                return f"{num:.2e}"
                
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {format_number(abs(avg_a_diff))} (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {format_number(abs(avg_new_diff))} (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        
        props_a = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=a_color)
        props_new = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=new_color)
        
        ax1.text(0.98, 0.95, a_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=a_color, bbox=props_a)
        ax1.text(0.98, 0.88, new_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=new_color, bbox=props_new)
        
        pd_name_title = ' '.join(word.capitalize() for word in pd_name.split('_')).split('(')[0].strip()
        if driver_name:
            pd_name_title += f" ({driver_name})"
        ax1.set_title(f'{pd_name_title} Utilisation: {self.name_a} vs {self.name_b} vs {new_data_name}', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Throughput', fontsize=12)
        
        if use_percent:
            ax1.set_ylabel('Percentage of Total Utilisation (%)', fontsize=12)
        else:
            ax1.set_ylabel('CPU Utilisation (cycles)', fontsize=12)
            
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        self._set_x_ticks(ax1, x)
        
        ax1.legend(loc='upper left', ncol=3, frameon=True)
        
        # Show max diff if requested
        if show_max_diff:
            a_max_diff_idx = np.argmax([abs(a_data[i] - b_data[i]) for i in range(len(iterations))])
            new_max_diff_idx = np.argmax([abs(new_data_values[i] - b_data[i]) for i in range(len(iterations))])
            
            if a_max_diff_idx == new_max_diff_idx:
                x_pos = x[a_max_diff_idx]
                ax1.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.15, color='purple')
                ax1.annotate(f"Max diff (both)", 
                        xy=(x_pos, y_limit * 0.95),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='purple',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="purple", alpha=0.8))
            else:
                # A max diff
                x_pos_a = x[a_max_diff_idx]
                ax1.axvspan(x_pos_a-0.5, x_pos_a+0.5, alpha=0.15, color='steelblue')
                ax1.annotate(f"Max diff ({self.name_a})", 
                        xy=(x_pos_a, y_limit * 0.95),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='steelblue',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="steelblue", alpha=0.8))
                
                # New max diff - staggered position
                x_pos_new = x[new_max_diff_idx]
                ax1.axvspan(x_pos_new-0.5, x_pos_new+0.5, alpha=0.15, color='forestgreen')
                ax1.annotate(f"Max diff ({new_data_name})", 
                        xy=(x_pos_new, y_limit * 0.88),
                        xytext=(0, 0),
                        textcoords="offset points",
                        ha='center', va='bottom',
                        fontsize=9, color='forestgreen',
                        bbox=dict(boxstyle="round,pad=0.1", fc="white", ec="forestgreen", alpha=0.8))
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        
        
    def plot_specific_pd_with_new_data_3(self, pd_name, new_data, new_data_name, save_path=None, figsize=(12, 8), dpi=100, show_max_diff=False, driver_name=None, use_percent=False):
        if pd_name not in self.pd_percent_a or pd_name not in self.pd_percent_b or pd_name not in new_data['pd_utilisation']:
            print(f"Error: PD '{pd_name}' not found in all datasets")
            return
            
        plt.figure(figsize=figsize, dpi=dpi)
        
        # Create the main plot and a smaller subplot for the difference indicators
        gs = plt.GridSpec(4, 1, height_ratios=[3, 1, 0.2, 0.8])
        ax1 = plt.subplot(gs[0])  # Main bar chart
        ax2 = plt.subplot(gs[1])  # Difference chart
        ax3 = plt.subplot(gs[3])  # Legend for difference indicators
        
        min_len = min(len(self.pd_percent_a[pd_name]), len(self.pd_percent_b[pd_name]), len(new_data['pd_utilisation'][pd_name]))
        iterations = range(1, min_len + 1)
        width = 0.25
        x = np.array(iterations)
        
        # Data preparation code (unchanged)
        if use_percent:
            a_data = self.pd_percent_a[pd_name][:min_len]
            b_data = self.pd_percent_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            total_util_new = [int(data['TotalUtilisation'], base_new) for data in new_data['total_utilisation']]
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            pd_percent_new = {}
            for pd, measurements in pd_data_new.items():
                pd_percent_new[pd] = np.array(measurements) / np.array(total_util_new) * 100
                
            new_data_values = pd_percent_new[pd_name][:min_len]
        else:
            a_data = self.pd_data_a[pd_name][:min_len]
            b_data = self.pd_data_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            new_data_values = pd_data_new[pd_name][:min_len]
        
        # Plot the bars in the main chart
        bars1 = ax1.bar(x - width, a_data, width, label=self.name_a, color='steelblue', alpha=0.8)
        bars2 = ax1.bar(x, b_data, width, label=self.name_b, color='darkorange', alpha=0.8)
        bars3 = ax1.bar(x + width, new_data_values, width, label=new_data_name, color='forestgreen', alpha=0.8)
        
        # Calculate differences relative to baseline (data_b)
        a_diffs = []
        new_diffs = []
        a_diff_pcts = []
        new_diff_pcts = []
        
        for i in range(len(iterations)):
            a_val = a_data[i]
            b_val = b_data[i]
            new_val = new_data_values[i]
            
            print(f"Iteration {i+1}: {pd_name} - {self.name_a}: {a_val}, {self.name_b}: {b_val}, {new_data_name}: {new_val}")
            
            a_diff = a_val - b_val
            a_diffs.append(a_diff)
            
            new_diff = new_val - b_val
            new_diffs.append(new_diff)
            
            # Calculate percentage differences for the heatmap
            a_diff_pct = (a_diff / b_val * 100) if b_val > 0 else 0
            new_diff_pct = (new_diff / b_val * 100) if b_val > 0 else 0
            
            a_diff_pcts.append(a_diff_pct)
            new_diff_pcts.append(new_diff_pct)
        
        # Calculate average differences
        avg_a_diff = np.mean([a_data[i] - b_data[i] for i in range(len(iterations))])
        avg_new_diff = np.mean([new_data_values[i] - b_data[i] for i in range(len(iterations))])
        
        # Setting up the difference visualization in the second subplot
        # Create two rows of colored blocks to represent differences
        bar_height = 0.3
        
        # Custom colormap for differences: red for positive, green for negative
        def get_diff_color(diff_pct):
            if diff_pct > 0:
                # Red scale for increases (0 to 25+%)
                intensity = min(abs(diff_pct) / 25.0, 1.0)
                return plt.cm.Reds(intensity)
            else:
                # Green scale for decreases (0 to 25+%)
                intensity = min(abs(diff_pct) / 25.0, 1.0)
                return plt.cm.Greens(intensity)
        
        # Plot difference bars for the first dataset
        for i in range(len(iterations)):
            ax2.bar(x[i], bar_height, width*2, bottom=0.35, color=get_diff_color(a_diff_pcts[i]), 
                    edgecolor='gray', linewidth=0.5, alpha=0.9)
            # Add text with exact percentage
            text_color = 'white' if abs(a_diff_pcts[i]) > 15 else 'black'
            ax2.text(x[i], 0.35 + bar_height/2, f"{a_diff_pcts[i]:.1f}%", 
                    ha='center', va='center', fontsize=8, color=text_color)
        
        # Plot difference bars for the new dataset
        for i in range(len(iterations)):
            ax2.bar(x[i], bar_height, width*2, bottom=0, color=get_diff_color(new_diff_pcts[i]), 
                    edgecolor='gray', linewidth=0.5, alpha=0.9)
            # Add text with exact percentage
            text_color = 'white' if abs(new_diff_pcts[i]) > 15 else 'black'
            ax2.text(x[i], bar_height/2, f"{new_diff_pcts[i]:.1f}%", 
                    ha='center', va='center', fontsize=8, color=text_color)
        
        # Add y-axis labels for the difference rows
        ax2.text(-0.5, 0.35 + bar_height/2, self.name_a, ha='right', va='center', fontsize=10)
        ax2.text(-0.5, bar_height/2, new_data_name, ha='right', va='center', fontsize=10)
        
        # Remove y-axis ticks and labels for the difference subplot
        ax2.set_yticks([])
        ax2.set_ylabel("Difference vs " + self.name_b, fontsize=10)
        ax2.set_title("Percentage Difference", fontsize=10)
        
        # Settings for difference subplot
        ax2.set_xlim(ax1.get_xlim())
        self._set_x_ticks(ax2, x)
        ax2.grid(True, linestyle='--', alpha=0.3, axis='x')
        
        # Create color scale legend for the difference indicator
        ax3.set_title("Difference Scale", fontsize=9)
        
        # Create color gradients
        gradient_width = 0.35
        pos_values = np.linspace(0, 25, 100)
        neg_values = np.linspace(-25, 0, 100)
        
        # Plot positive gradient (red)
        for i, v in enumerate(pos_values[:-1]):
            ax3.fill_between([v, pos_values[i+1]], [0, 0], [gradient_width, gradient_width], 
                            color=plt.cm.Reds((v/25.0)))
        
        # Plot negative gradient (green)
        for i, v in enumerate(neg_values[:-1]):
            ax3.fill_between([v, neg_values[i+1]], [0, 0], [gradient_width, gradient_width], 
                            color=plt.cm.Greens((abs(v)/25.0)))
        
        # Add tick marks to the color scale
        ax3.set_yticks([])
        ax3.set_xticks([-25, -15, -5, 0, 5, 15, 25])
        ax3.set_xlim(-25, 25)
        ax3.set_xlabel("Change in % vs " + self.name_b, fontsize=8)
        
        # Add overall difference indicators
        epsilon = 1e-10
        a_rel_diffs_no_zeros = [max(abs((a_data[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        new_rel_diffs_no_zeros = [max(abs((new_data_values[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        
        avg_a_rel_diff = stats.gmean(a_rel_diffs_no_zeros)
        avg_new_rel_diff = stats.gmean(new_rel_diffs_no_zeros)
        
        a_arrow = "↑" if avg_a_diff > 0 else "↓"
        a_color = "red" if avg_a_diff > 0 else "green"
        a_sign = "+" if avg_a_diff > 0 else "-"
        
        new_arrow = "↑" if avg_new_diff > 0 else "↓"
        new_color = "red" if avg_new_diff > 0 else "green"
        new_sign = "+" if avg_new_diff > 0 else "-"
        
        if use_percent:
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {abs(avg_a_diff):.1f}% (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {abs(avg_new_diff):.1f}% (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        else:
            def format_number(num):
                if abs(num) == 0:
                    return "0"
                return f"{num:.2e}"
                
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {format_number(abs(avg_a_diff))} (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {format_number(abs(avg_new_diff))} (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        
        props_a = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=a_color)
        props_new = dict(boxstyle='round', facecolor='white', alpha=0.9, edgecolor=new_color)
        
        ax1.text(0.98, 0.95, a_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=a_color, bbox=props_a)
        ax1.text(0.98, 0.88, new_diff_text, transform=ax1.transAxes, fontsize=10,
            verticalalignment='top', horizontalalignment='right', color=new_color, bbox=props_new)
        
        # Main plot settings
        pd_name_title = ' '.join(word.capitalize() for word in pd_name.split('_')).split('(')[0].strip()
        if driver_name:
            pd_name_title += f" ({driver_name})"
        ax1.set_title(f'{pd_name_title} Utilisation: {self.name_a} vs {self.name_b} vs {new_data_name}', fontsize=14, fontweight='bold')
        
        if use_percent:
            ax1.set_ylabel('Percentage of Total Utilisation (%)', fontsize=12)
        else:
            ax1.set_ylabel('CPU Utilisation (cycles)', fontsize=12)
            
        ax1.grid(True, linestyle='--', alpha=0.3, axis='y')
        
        self._set_x_ticks(ax1, x)
        ax1.set_xlabel('')  # Remove x-label from main plot
        
        ax1.legend(loc='upper left', ncol=3, frameon=True)
        
        # Handle max diff highlighting in the main plot
        if show_max_diff:
            a_max_diff_idx = np.argmax([abs(a_data[i] - b_data[i]) for i in range(len(iterations))])
            new_max_diff_idx = np.argmax([abs(new_data_values[i] - b_data[i]) for i in range(len(iterations))])
            
            if a_max_diff_idx == new_max_diff_idx:
                x_pos = x[a_max_diff_idx]
                ax1.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.15, color='purple')
                ax2.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.3, color='purple')
            else:
                # A max diff
                x_pos_a = x[a_max_diff_idx]
                ax1.axvspan(x_pos_a-0.5, x_pos_a+0.5, alpha=0.15, color='steelblue')
                ax2.axvspan(x_pos_a-0.5, x_pos_a+0.5, alpha=0.3, color='steelblue')
                
                # New max diff
                x_pos_new = x[new_max_diff_idx]
                ax1.axvspan(x_pos_new-0.5, x_pos_new+0.5, alpha=0.15, color='forestgreen')
                ax2.axvspan(x_pos_new-0.5, x_pos_new+0.5, alpha=0.3, color='forestgreen')
        
        # Final x-axis label at the bottom
        ax2.set_xlabel('Throughput', fontsize=12)
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, bbox_inches='tight')
        
        plt.show()
        
    def plot_specific_pd_with_new_data_4(self, pd_name, new_data, new_data_name, save_path=None, figsize=(12, 7), dpi=120, show_max_diff=False, driver_name=None, use_percent=False):
        if pd_name not in self.pd_percent_a or pd_name not in self.pd_percent_b or pd_name not in new_data['pd_utilisation']:
            print(f"Error: PD '{pd_name}' not found in all datasets")
            return
        
        # Create figure with modern style
        plt.style.use('seaborn-v0_8-pastel')
        fig, ax1 = plt.subplots(figsize=figsize, dpi=dpi)
        
        # Prepare data
        min_len = min(len(self.pd_percent_a[pd_name]), len(self.pd_percent_b[pd_name]), len(new_data['pd_utilisation'][pd_name]))
        iterations = range(1, min_len + 1)
        width = 0.25
        x = np.array(iterations)
        
        # Process data (unchanged)
        if use_percent:
            a_data = self.pd_percent_a[pd_name][:min_len]
            b_data = self.pd_percent_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            total_util_new = [int(data['TotalUtilisation'], base_new) for data in new_data['total_utilisation']]
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            pd_percent_new = {}
            for pd, measurements in pd_data_new.items():
                pd_percent_new[pd] = np.array(measurements) / np.array(total_util_new) * 100
                
            new_data_values = pd_percent_new[pd_name][:min_len]
        else:
            a_data = self.pd_data_a[pd_name][:min_len]
            b_data = self.pd_data_b[pd_name][:min_len]
            
            is_hex_format = lambda value: any(c in "abcdefABCDEF" for c in str(value))
            use_hex_new = False
            samples_new = []
            if new_data['total_utilisation']:
                samples_new.append(new_data['total_utilisation'][0]['TotalUtilisation'])
            
            for sample in samples_new:
                if is_hex_format(sample):
                    use_hex_new = True
                    break
            
            base_new = 16 if use_hex_new else 10
            
            pd_data_new = {pd: [int(m['TotalUtilisation'], base_new) for m in measurements]
                        for pd, measurements in new_data['pd_utilisation'].items()}
            
            new_data_values = pd_data_new[pd_name][:min_len]
        
        # Calculate differences and percentages
        a_diffs = []
        new_diffs = []
        a_diff_pcts = []
        new_diff_pcts = []
        
        for i in range(len(iterations)):
            a_val = a_data[i]
            b_val = b_data[i]
            new_val = new_data_values[i]
            
            print(f"Iteration {i+1}: {pd_name} - {self.name_a}: {a_val}, {self.name_b}: {b_val}, {new_data_name}: {new_val}")
            
            a_diff = a_val - b_val
            a_diffs.append(a_diff)
            
            new_diff = new_val - b_val
            new_diffs.append(new_diff)
            
            # Calculate percentage differences
            a_diff_pct = (a_diff / b_val * 100) if b_val > 0 else 0
            new_diff_pct = (new_diff / b_val * 100) if b_val > 0 else 0
            
            a_diff_pcts.append(a_diff_pct)
            new_diff_pcts.append(new_diff_pct)
        
        # Vibrant presentation colors
        colors = ['#2196F3', '#FF9800', '#4CAF50']  # Blue, Orange, Green - vibrant and distinct
        
        # Plot bars with vibrant colors
        bars1 = ax1.bar(x - width, a_data, width, label=self.name_a, color=colors[0], alpha=0.85)
        bars2 = ax1.bar(x, b_data, width, label=self.name_b, color=colors[1], alpha=0.85)
        bars3 = ax1.bar(x + width, new_data_values, width, label=new_data_name, color=colors[2], alpha=0.85)
        
        # Calculate max height for arrow positioning
        max_height = max(max(a_data), max(b_data), max(new_data_values))
        y_limit = max_height * 1.2  # Leaving space for annotations
        ax1.set_ylim(0, y_limit)
        
        # Add arrows and percentage labels using visually appealing arrows and colors
        for i, (bar1, bar2, bar3) in enumerate(zip(bars1, bars2, bars3)):
            # Get heights
            height1 = bar1.get_height()
            height2 = bar2.get_height()
            height3 = bar3.get_height()
            
            # First comparison - only show if significant (>3%) to avoid clutter
            if abs(a_diff_pcts[i]) > 3:
                # Determine direction and color of arrow
                if a_diff_pcts[i] > 0:
                    arrow_color = '#E53935'  # Red for increase
                    arrow_style = '↑'
                else:
                    arrow_color = '#43A047'  # Green for decrease
                    arrow_style = '↓'
                
                # Format percentage text
                pct_text = f"{arrow_style} {abs(a_diff_pcts[i]):.1f}%"
                
                # Draw a curved arrow connecting bars 1 and 2
                # Use the bar positions to calculate the arrow start and end points
                x_start = bar1.get_x() + bar1.get_width()/2
                y_start = max(height1, height2) + (y_limit * 0.02)
                
                x_end = bar2.get_x() + bar2.get_width()/2
                y_end = y_start
                
                # Position the arrow at staggered heights based on index
                # This ensures even bars don't overlap their annotations
                y_offset = y_limit * 0.05 * (1 + (i % 3) * 0.3)
                
                # Draw the connecting path
                ax1.annotate('', 
                        xy=(x_end, y_end + y_offset * 0.8), 
                        xytext=(x_start, y_start + y_offset),
                        arrowprops=dict(arrowstyle='->', 
                                        color=arrow_color, 
                                        linewidth=1.5,
                                        connectionstyle="arc3,rad=.3"))
                
                # Add the percentage text in the middle of the arc
                mid_x = (x_start + x_end) / 2 - 0.05  # Slight adjustment to center on the arc
                mid_y = y_start + y_offset * 0.9
                
                ax1.text(mid_x, mid_y, pct_text,
                        ha='center', va='center',
                        fontsize=10, color=arrow_color,
                        fontweight='bold')
            
            # Second comparison - only show if significant (>3%)
            if abs(new_diff_pcts[i]) > 3:
                # Determine direction and color of arrow
                if new_diff_pcts[i] > 0:
                    arrow_color = '#E53935'  # Red
                    arrow_style = '↑'
                else:
                    arrow_color = '#43A047'  # Green
                    arrow_style = '↓'
                
                # Format percentage text
                pct_text = f"{arrow_style} {abs(new_diff_pcts[i]):.1f}%"
                
                # Draw a curved arrow connecting bars 3 and 2
                x_start = bar3.get_x() + bar3.get_width()/2
                y_start = max(height3, height2) + (y_limit * 0.02)
                
                x_end = bar2.get_x() + bar2.get_width()/2
                y_end = y_start
                
                # Position at staggered heights based on index
                # Use a different stagger pattern from the first comparison
                y_offset = y_limit * 0.05 * (1 + ((i + 1) % 3) * 0.3)
                
                # Draw connecting path
                ax1.annotate('', 
                        xy=(x_end, y_end + y_offset * 0.8), 
                        xytext=(x_start, y_start + y_offset),
                        arrowprops=dict(arrowstyle='->', 
                                        color=arrow_color, 
                                        linewidth=1.5,
                                        connectionstyle="arc3,rad=-.3"))
                
                # Add percentage text
                mid_x = (x_start + x_end) / 2 + 0.05  # Slight adjustment
                mid_y = y_start + y_offset * 0.9
                
                ax1.text(mid_x, mid_y, pct_text,
                        ha='center', va='center',
                        fontsize=10, color=arrow_color,
                        fontweight='bold')
        
        # Calculate average differences
        avg_a_diff = np.mean([a_data[i] - b_data[i] for i in range(len(iterations))])
        avg_new_diff = np.mean([new_data_values[i] - b_data[i] for i in range(len(iterations))])
        
        # Calculate geometric mean for relative differences
        epsilon = 1e-10
        a_rel_diffs_no_zeros = [max(abs((a_data[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        new_rel_diffs_no_zeros = [max(abs((new_data_values[i] - b_data[i]) / b_data[i]) * 100 if b_data[i] > 0 else 0, epsilon) for i in range(len(iterations))]
        
        avg_a_rel_diff = stats.gmean(a_rel_diffs_no_zeros)
        avg_new_rel_diff = stats.gmean(new_rel_diffs_no_zeros)
        
        # Determine colors and arrows for overall differences
        a_arrow = "↑" if avg_a_diff > 0 else "↓"
        a_color = "#E53935" if avg_a_diff > 0 else "#43A047"
        a_sign = "+" if avg_a_diff > 0 else "-"
        
        new_arrow = "↑" if avg_new_diff > 0 else "↓"
        new_color = "#E53935" if avg_new_diff > 0 else "#43A047"
        new_sign = "+" if avg_new_diff > 0 else "-"
        
        if use_percent:
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {abs(avg_a_diff):.1f}% (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {abs(avg_new_diff):.1f}% (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        else:
            def format_number(num):
                if abs(num) < 0.001:
                    return "0"
                return f"{num:.2e}"
                
            a_diff_text = f"{self.name_a} Δ: {a_arrow} {format_number(abs(avg_a_diff))} (Geom. {a_sign}{avg_a_rel_diff:.1f}%)"
            new_diff_text = f"{new_data_name} Δ: {new_arrow} {format_number(abs(avg_new_diff))} (Geom. {new_sign}{avg_new_rel_diff:.1f}%)"
        
        # Add summary box with gradient background
        from matplotlib.patches import FancyBboxPatch
        
        # Create a fancy box for the summary
        props = dict(boxstyle='round,pad=0.6', facecolor='#F5F5F5', alpha=0.9, edgecolor='#BDBDBD')
        
        # Place the summary box in the upper right
        summary_text = f"Average Differences vs {self.name_b}:\n" + \
                    f"{a_diff_text}\n" + \
                    f"{new_diff_text}"
        
        ax1.text(0.97, 0.97, summary_text, transform=ax1.transAxes,
                fontsize=10, verticalalignment='top', horizontalalignment='right',
                bbox=props)
        
        # Set title and labels with presentation styling
        pd_name_title = ' '.join(word.capitalize() for word in pd_name.split('_')).split('(')[0].strip()
        if driver_name:
            pd_name_title += f" ({driver_name})"
        
        # Add title with background highlight for emphasis
        title = f'{pd_name_title} Utilisation: {self.name_a} vs {self.name_b} vs {new_data_name}'
        ax1.set_title(title, fontsize=14, fontweight='bold', pad=15)
        
        # Add a subtle title background
        title_patch = plt.Rectangle((0, 1.01), 1, 0.08, transform=ax1.transAxes,
                                facecolor='#E3F2FD', alpha=0.6, zorder=-1)
        ax1.add_patch(title_patch)
        
        # Set axis labels
        if use_percent:
            ax1.set_ylabel('Percentage of Total Utilisation (%)', fontsize=12, fontweight='bold')
        else:
            ax1.set_ylabel('CPU Utilisation (cycles)', fontsize=12, fontweight='bold')
            
        ax1.set_xlabel('Throughput', fontsize=12, fontweight='bold')
        
        # Presentation-style grid and background
        ax1.grid(axis='y', linestyle='--', alpha=0.3)
        ax1.set_axisbelow(True)  # Put grid behind bars
        
        # Set x-ticks
        self._set_x_ticks(ax1, x)
        
        # Visually appealing legend
        leg = ax1.legend(ncol=3, loc='upper left', frameon=True, fancybox=True, 
                        framealpha=0.85, edgecolor='#BDBDBD', fontsize=10)
        
        # Handle max diff highlighting with more visual emphasis for presentation
        if show_max_diff:
            a_max_diff_idx = np.argmax([abs(a_data[i] - b_data[i]) for i in range(len(iterations))])
            new_max_diff_idx = np.argmax([abs(new_data_values[i] - b_data[i]) for i in range(len(iterations))])
            
            if a_max_diff_idx == new_max_diff_idx:
                x_pos = x[a_max_diff_idx]
                # Use gradient highlighting
                ax1.axvspan(x_pos-0.5, x_pos+0.5, alpha=0.15, color='#673AB7')
                ax1.text(x_pos, ax1.get_ylim()[1] * 0.98, "Max Diff (Both)", 
                        ha='center', va='top', fontsize=10, fontweight='bold', color='#673AB7',
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec="#673AB7", alpha=0.9))
            else:
                # A max diff with vibrant highlighting
                x_pos_a = x[a_max_diff_idx]
                ax1.axvspan(x_pos_a-0.5, x_pos_a+0.5, alpha=0.15, color=colors[0])
                ax1.text(x_pos_a, ax1.get_ylim()[1] * 0.98, f"Max Diff ({self.name_a})", 
                        ha='center', va='top', fontsize=9, fontweight='bold', color=colors[0],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[0], alpha=0.9))
                
                # New max diff with vibrant highlighting
                x_pos_new = x[new_max_diff_idx]
                ax1.axvspan(x_pos_new-0.5, x_pos_new+0.5, alpha=0.15, color=colors[2])
                ax1.text(x_pos_new, ax1.get_ylim()[1] * 0.90, f"Max Diff ({new_data_name})", 
                        ha='center', va='top', fontsize=9, fontweight='bold', color=colors[2],
                        bbox=dict(boxstyle="round,pad=0.3", fc="white", ec=colors[2], alpha=0.9))
        
        # Final touch: add subtle curved background for aesthetic appeal
        gradient = np.linspace(0, 1, 100).reshape(1, -1)
        gradient = np.vstack((gradient, gradient))
        extent = [0, 1, 0, 1]
        ax1.imshow(gradient, aspect='auto', extent=extent, 
                origin='lower', alpha=0.05, cmap='Blues',
                transform=ax1.transAxes, zorder=-10)
        
        # Tight layout for presentation
        plt.tight_layout()
        
        # Save and show
        if save_path:
            plt.savefig(save_path, bbox_inches='tight', dpi=dpi)
        
        plt.show()