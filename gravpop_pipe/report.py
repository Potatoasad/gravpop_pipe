from dataclasses import dataclass, field
from typing import List
import matplotlib
from gravpop import *
from .parser import *
import numpy as np
import matplotlib.pyplot as plt

@dataclass
class Report:
    parsers : List[Any]
    names : List[str]
    color_assigner : Optional[Union[matplotlib.colors.ListedColormap, np.ndarray]] = field(default_factory=lambda : matplotlib.cm.tab10)
    
    def __post_init__(self):
        if isinstance(self.color_assigner, matplotlib.colors.ListedColormap):
            self.color_assigner =  self.color_assigner(range(len(self.parsers)))
            
        plt.rcParams['text.usetex'] = True
    
    def create_mass_plot(self, height_ratios=[1.27, 1], figsize=(8,6), dpi=200):
        fig = plt.figure(layout="constrained", figsize=figsize, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=height_ratios, figure=fig)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        colors = self.color_assigner
        for i in range(len(self.parsers)):
            self.parsers[i].hyper_posterior.mass_plot.mass_marginal_plot(log_lower=-6, ax=ax1, label=self.names[i], color=colors[i])
        
        ax1.legend()
        
        for i in range(len(self.parsers)):
            self.parsers[i].hyper_posterior.mass_plot.mass_ratio_marginal_plot(aspect=0.3, log_lower=-2, ax=ax2, label=self.names[i], color=colors[i])
        
        ax2.legend()
        return fig
    
    def create_redshift_plot(self, dpi=200):
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(1, dpi=dpi)
        colors = self.color_assigner
        for i in range(len(self.parsers)):
            self.parsers[i].hyper_posterior.redshift_plot.plot_model(ax=ax, log_lower=-3, label=self.names[i], color=colors[i])
        ax.legend()
        return fig
    
    def get_max_prob_spin(self, result, grid, axis=1):
        return jax.scipy.integrate.trapezoid(result, grid, axis=1).max()
    
    def create_spin_magnitude_plot(self, dpi=200, figsize=(8,6), aspect=0.4):
        import matplotlib.pyplot as plt
        import matplotlib

        fig = plt.figure(layout="constrained", figsize=figsize, dpi=dpi)
        gs = matplotlib.gridspec.GridSpec(2, 1, height_ratios=[1, 1], figure=fig)

        ax1 = fig.add_subplot(gs[0])
        ax2 = fig.add_subplot(gs[1])

        colors = self.color_assigner
        
        max_1s = []
        
        for i in range(len(self.parsers)):
            self.parsers[i].hyper_posterior.spin_magnitude_plot.spin_1_marginal_plot(aspect=0.4, ax=ax1, label=self.names[i], color=colors[i])
            max_1s.append(self.get_max_prob_spin(self.parsers[i].hyper_posterior.spin_magnitude_plot.result, 
                                                self.parsers[i].hyper_posterior.spin_magnitude_plot.spin_1_grid,
                                                axis=1))
        
        
        ax1.legend()

        max_2s = []
        for i in range(len(self.parsers)):
            self.parsers[i].hyper_posterior.spin_magnitude_plot.spin_2_marginal_plot(aspect=0.4, ax=ax2, label=self.names[i], color=colors[i])
            max_2s.append(self.get_max_prob_spin(self.parsers[i].hyper_posterior.spin_magnitude_plot.result, 
                                                self.parsers[i].hyper_posterior.spin_magnitude_plot.spin_2_grid,
                                                axis=2))
                          
            
        ax2.legend()
        ax1.set_aspect(aspect / max(max_1s))
        ax2.set_aspect(aspect / max(max_2s))

        plt.legend()
        return fig
    
    def create_corner(self, colnames=None):
        import corner
        colnames = colnames or self.parsers[0].hyper_parameters
        latex_colnames = [self.parsers[0].latex_names[x] for x in colnames]
        fig = corner.corner(self.parsers[0].hyper_posterior.posterior[colnames].values, 
                            labels=latex_colnames, color=self.color_assigner[0], show_titles=True)
        
        for i in range(1,len(self.parsers)):
            corner.corner(self.parsers[i].hyper_posterior.posterior[colnames].values, fig=fig, labels=latex_colnames, color=self.color_assigner[i])
        
        import matplotlib.lines as mlines
        the_lines = []
        for i in range(len(self.parsers)):
            a_line = mlines.Line2D([], [], color=self.color_assigner[i], label=self.names[i])
            the_lines.append(a_line)

        plt.legend(handles=the_lines, bbox_to_anchor=(0., 1.5, 1., .0), loc=4, fontsize="20")
        return fig
    
    def create_mass_redshift_corner(self):
        mass_vars = convert_to_list(self.parsers[0].config_dict['Variables']['Population']['mass'])
        redshift_vars = convert_to_list(self.parsers[0].config_dict['Variables']['Population']['redshift'])
        cols = mass_vars + redshift_vars
        return self.create_corner(colnames=cols)
    
    def create_corner_for_models(self, model_list=['mass', 'redshift']):
        cols = []
        for model in model_list:
            cols += convert_to_list(self.parsers[0].config_dict['Variables']['Population'][model])
        return self.create_corner(colnames=cols)
            
    def save_image(self, filename=None):
        from matplotlib.backends.backend_pdf import PdfPages
        if filename is None:
            if 'plots' in self.config_dict['Output']:
                output_plot_location = self.config_dict['Output']['plots']
            else:
                output_plot_location = "./output.pdf"
        p = PdfPages(filename) 
        
        #figs = [plt.figure(n) for n in fig_nums] 
        figs = []
        figs.append(self.create_mass_plot())
        figs.append(self.create_redshift_plot())
        figs.append(self.create_spin_magnitude_plot())
        figs.append(self.create_corner_for_models(['mass', 'redshift']))
        figs.append(self.create_corner_for_models(['spin_magnitude']))
        figs.append(self.create_corner())

        # iterating over the numbers in list 
        for fig in figs:
            # and saving the files 
            fig.savefig(p, format='pdf')  

        # close the object 
        p.close()   