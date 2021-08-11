"""
A utility file with functions to make plots, to make analysis files more
readable and also to make them easier to make.

author: Aria Radick
date: 6/1/20
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.patches import Rectangle
import matplotlib.gridspec as gridspec
from pyQEdark.constants import ckms
from pyQEdark.crystaldme import Crystal_DMe


"""                             v^2 f(v) Class                              """


class v2f_figure:

    """
    An object to simplify plotting the v^2 f(v) plots for velocity distribution
    functions.

    Inputs :
    fig_ : The figure you wish to apply this plot to.
    Vel : The list of velocities to plot.
    Title : The title to draw on the figure.
    plot_shm : This decides whether or not to plot the fiducial shm line. If
               this is true, it also uses the shm fiducial line as the
               comparison in the ratio plot. If plot_shm is set to False, you
               must provide the following two variables in kwargs:
        fidvals : A list of the v2f values you wish to compare against. This
                  will also be plotted in ax1 instead of the shm fiducial.
        label : A label for your fiducial line.
    """

    def __init__(self, fig_, Vel, Title, plot_shm=True, **kwargs):

        params = {'text.usetex' : True, \
                  'font.size' : 14, \
                  'font.family' : 'cmr10', \
                  'figure.autolayout': True}
        plt.rcParams.update(params)
        plt.rcParams['axes.unicode_minus']=False
        plt.rcParams['axes.labelsize']=16

        self.diffsch = self.ratio
        self.difftxt = 'ratio'

        self.Vel = Vel

        self.ylim1 = (-1e-5, 5e-4)
        self.ylim2 = (0,2)
        self.ytck2 = [.25, .5, .75, 1., 1.25, 1.5, 1.75]
        self.ylab2 = [None, .5, None, 1, None, 1.5, None]
        self.isLeg = False

        self.set_params(**kwargs)

        self.gs = fig_.add_gridspec(3,2)
        self.gs.update(hspace=0.0)

        # add first subplot, in the top 2x2 square of the grid:
        self.ax1 = fig_.add_subplot(self.gs[0:2, :], xlim=(Vel[0], Vel[-1]),
                                    ylim=self.ylim1)
        # add second subplot, with shared x-axis, in the bottom 1x2:
        self.ax2 = fig_.add_subplot(self.gs[2,:], ylim=self.ylim2,
                                    sharex=self.ax1)

        # set up the axes:
        self.ax2.set_yticks(self.ytck2)
        self.ax2.set_yticklabels(self.ylab2)
        self.ax2.grid(True, axis='y')
        plt.setp(self.ax1.get_xticklabels(), visible=False)

        self.ax1.tick_params(direction='in', top=True, right=True) #ticks inside
        self.ax2.tick_params(direction='in', top=True, right=True)

        self.lines = []
        self.cbands = []

        if plot_shm:
            from utils.velocity_dists import f_SHM
            from utils.astroparams_kms import v0fid, vescfid
            self.fidvals = f_SHM(self.Vel, [v0fid, vescfid])
            self.lines.append( self.ax1.plot(self.Vel, self.fidvals, color='k',
                                             label='SHM fiducial' )[0] )
            self.ax2.plot(self.Vel, np.ones_like(self.Vel), c='k')

        else:
            self.fidvals = fidvals
            self.lines.append( self.ax1.plot(self.Vel, self.fidvals, color='k',
                                             label=label )[0] )
            self.ax2.plot(self.Vel, np.ones_like(self.Vel), c='k')

        self.set_legend()

        self.ax2.set_xlabel(r'$ v $ [km/s]')
        self.ax1.set_ylabel(r'$v^2 f(v)$ [(km/s)$^{-1}$]')
        self.ax2.set_ylabel('ratio', fontsize=10)
        self.ax1.set_title(Title)

    def set_params(self, **kwargs):
        allowed_keys = {'ylim1', 'ylim2', 'ytck2', 'ylab2', 'diffsch',
                        'difftxt'}
        self.__dict__.update((k,v) for k,v in kwargs.items() if k in allowed_keys)

    def ratio(self, _vals, _fid):
        return _vals/_fid

    def set_legend(self):

        if self.isLeg:
            self.first_legend.remove()
            self.second_legend.remove()

        self.first_legend = self.ax1.legend(handles=self.lines,
                                            loc='upper left', frameon=False,
                                            fontsize=10)
        self.second_legend = self.ax1.legend(handles=self.cbands,
                                             loc='upper right', frameon=False,
                                             fontsize=10)

        self.ax1.add_artist(self.first_legend)
        self.ax1.add_artist(self.second_legend)

        self.isLeg = True

    def add_line(self, v2f_vals, label=None, c=None, ls='solid'):
        """
        Adds a line to your figure onto both axes.
        On ax1, this will plot the vals you enter into v2f_vals. On ax2, this
        will plot diffsch(v2f_vals, fidvals) where fidvals will be either the
        shm fiducial values or your chosen fidvals entered in upon creation
        of this object.
        label : the label to show in the legend.
        c : the color of the line.
        ls : the linestyle (see matplotlib docs for options).
        """

        self.lines.append( self.ax1.plot( self.Vel, v2f_vals, color=c,
                                          label=label, ls=ls )[0] )
        self.ax2.plot( self.Vel, self.diffsch(v2f_vals, self.fidvals),
                       c=c, ls=ls )

        self.set_legend()

    def add_band(self, vals_low, vals_high, c, label=None, hatch=None):
        """
        Adds a colorband to your figure onto both axes.
        On ax1, this will color in the area between vals_low and vals_high with
        the color set by 'c' and the hatch set by 'hatch'.
        label : the label to show in the legend.
        """

        self.cbands.append( self.ax1.fill_between(self.Vel, vals_low, vals_high,
                            color=c, hatch=hatch, alpha=0.3, label=label) )
        self.ax2.fill_between(self.Vel, self.diffsch(vals_low, self.fidvals),
                              self.diffsch(vals_high, self.fidvals), color=c,
                              hatch=hatch, alpha=0.3)

        self.set_legend()


"""                                   eta class                              """


class eta_figure:

    """
    An object to simplify plotting the eta(vmin) plots for velocity distribution
    functions.

    Inputs :
    fig_ : The figure you wish to apply this plot to.
    Vmin : The list of minimum velocities to plot.
    Title : The title to draw on the figure.
    plot_shm : This decides whether or not to plot the fiducial shm line. If
               this is true, it also uses the shm fiducial line as the
               comparison in the ratio plot. If plot_shm is set to False, you
               must provide the following two variables in kwargs:
        fidvals : A list of the eta values you wish to compare against. This
                  will also be plotted in ax1 instead of the shm fiducial.
        label : A label for your fiducial line.
    """

    def __init__(self, fig_, Vmin, Title, plot_shm=True, **kwargs):

        params = {'text.usetex' : True, \
                  'font.size' : 14, \
                  'font.family' : 'cmr10', \
                  'figure.autolayout': True}
        plt.rcParams.update(params)
        plt.rcParams['axes.unicode_minus']=False
        plt.rcParams['axes.labelsize']=16

        self.diffsch = self.ratio
        self.difftxt = 'ratio'

        self.Vmin = Vmin
        self.ylim1 = (1e-6, 1e-2)
        self.ytck1 = [1e-5, 1e-4, 1e-3]
        self.ylab1 = [1e-5, 1e-4, 1e-3]
        self.ylim2 = (1e-1, 1e1)
        self.ytck2 = [2e-1, 3e-1, 5e-1, 1, 2, 3, 5]
        self.ylab2 = [None, .3, None, 1, None, 3, None]
        self.isLeg = False

        self.set_params(**kwargs)

        self.gs = fig_.add_gridspec(3,2)
        self.gs.update(hspace=0.0)

        # add first subplot, in the top 2x2 square of the grid:
        self.ax1 = fig_.add_subplot(self.gs[0:2, :], xlim=(Vmin[0], Vmin[-1]),
                                    ylim=self.ylim1)
        # add second subplot, with shared x-axis, in the bottom 1x2:
        self.ax2 = fig_.add_subplot(self.gs[2,:], ylim=self.ylim2,
                                    sharex=self.ax1)

        # set up the axes:
        self.ax1.set_yscale('log')
        self.ax1.set_yticks(self.ytck1)
        self.ax1.set_yticklabels(self.ylab1)
        self.ax2.set_yscale('log')
        self.ax2.set_yticks(self.ytck2)
        self.ax2.set_yticklabels(self.ylab2)
        self.ax2.grid(True, axis='y')
        plt.setp(self.ax1.get_xticklabels(), visible=False)

        self.ax1.tick_params(direction='in', top=True, right=True) #ticks inside
        self.ax2.tick_params(direction='in', top=True, right=True)

        self.lines = []
        self.cbands = []

        if plot_shm:
            from utils.etas import etaSHM
            from utils.astroparams_kms import v0fid, vEfid, vescfid
            self.fidvals = etaSHM(self.Vmin, [v0fid, vEfid, vescfid])
            self.lines.append( self.ax1.plot(self.Vmin, self.fidvals, color='k',
                                             label='SHM fiducial' )[0] )
            self.ax2.plot(self.Vmin, np.ones_like(self.Vmin), c='k')

        else:
            self.fidvals = fidvals
            self.lines.append( self.ax1.plot(self.Vmin, self.fidvals, color='k',
                                             label=label )[0] )
            self.ax2.plot(self.Vmin, np.ones_like(self.Vmin), c='k')

        self.set_legend()

        self.ax2.set_xlabel(r'$ v_{\rm min}$ [km/s]')
        self.ax1.set_ylabel(r'$\eta(v_{\rm min})$ [(km/s)$^{-1}$] ')
        self.ax2.set_ylabel('ratio', fontsize=10)
        self.ax1.set_title(Title)

    def set_params(self, **kwargs):
        allowed_keys = {'ylim1', 'ytck1', 'ylab1', 'ylim2', 'ytck2', 'ylab2',
                        'diffsch', 'difftxt'}
        self.__dict__.update((k,v) for k,v in kwargs.items() if k in allowed_keys)

    def ratio(self, _vals, _fid):
        return _vals/_fid

    def set_legend(self):

        if self.isLeg:
            self.first_legend.remove()
            self.second_legend.remove()

        self.first_legend = self.ax1.legend(handles=self.lines,
                                            loc='lower left', frameon=False,
                                            fontsize=10)
        self.second_legend = self.ax1.legend(handles=self.cbands,
                                             loc='upper right', frameon=False,
                                             fontsize=10)

        self.ax1.add_artist(self.first_legend)
        self.ax1.add_artist(self.second_legend)

        self.isLeg = True

    def add_line(self, eta_vals, label=None, c=None, ls=None):
        """
        Adds a line to your figure onto both axes.
        On ax1, this will plot the vals you enter into v2f_vals. On ax2, this
        will plot diffsch(v2f_vals, fidvals) where fidvals will be either the
        shm fiducial values or your chosen fidvals entered in upon creation
        of this object.
        label : the label to show in the legend.
        c : the color of the line.
        ls : the linestyle (see matplotlib docs for options).
        """

        self.lines.append( self.ax1.plot( self.Vmin, eta_vals, color=c,
                                          label=label, ls=ls )[0] )
        self.ax2.plot( self.Vmin, self.diffsch(eta_vals, self.fidvals),
                       c=c, ls=ls )

        self.set_legend()

    def add_band(self, vals_low, vals_high, c, label=None, hatch=None):
        """
        Adds a colorband to your figure onto both axes.
        On ax1, this will color in the area between vals_low and vals_high with
        the color set by 'c' and the hatch set by 'hatch'.
        label : the label to show in the legend.
        """

        self.cbands.append(self.ax1.fill_between(self.Vmin, vals_low, vals_high,
                           color=c, hatch=hatch, alpha=0.3, label=label))
        self.ax2.fill_between(self.Vmin, self.diffsch(vals_low, self.fidvals),
                              self.diffsch(vals_high, self.fidvals), color=c,
                              hatch=hatch, alpha=0.3)

        self.set_legend()


"""                                 Rate Class                              """


class Rate_figure:

    """
    An object that simplifies plotting of rates. Plots a 2x2 grid with the
    columns being defined by FDMn and the rows being defined by m_chi.

    Inputs :
    fig_ : the figure you wish to add these plots to.
    Title : the suptitle for the figure.
    CrysObj : An object of type Crystal_DMe.
    mX : A list of masses you wish to consider in eV (2 is best supported)
    fdmn : A list of FDMn values you wish to consider (2 is best supported)
    plot_shm : This decides whether or not to plot the fiducial shm line. If
               this is set to False, then the class will draw a line defined
               by your CrysObj's current parameters.
    """

    def __init__(self, fig_, Title, CrysObj, mX, fdmn=[0,2],
                 plot_shm=True, **kwargs):

        params = {'text.usetex' : True, \
                  'font.size' : 14, \
                  'font.family' : 'cmr10', \
                  'figure.autolayout': True}
        plt.rcParams.update(params)
        plt.rcParams['axes.unicode_minus']=False
        plt.rcParams['axes.labelsize']=16

        self.fig = fig_

        self.ylim = [(1e-6, 1e6), (1e-4, 1e4)]
        self.ytck = [ [1e-5, 1e-4, 1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3, 1e4,
                       1e5],
                      [1e-3, 1e-2, 1e-1, 1e0, 1e1, 1e2, 1e3] ]
        self.if_ylab = [ True, False ]
        self.ylab = [ [r'$10^{-5}$', None, r'$10^{-3}$', None, r'$10^{-1}$',
                       None, r'$10^1$', None, r'$10^{3}$', None, r'$10^{5}$'] ]

        self.N_bins = 12

        first_time_keys = {'N_bins'}
        self.__dict__.update((k,v) for k,v in kwargs.items() \
                             if k in first_time_keys)

        self.set_params(**kwargs)

        self.material = CrysObj.material
        self.evals = CrysObj.get_evals(Ne_max=self.N_bins)
        xlims = (self.evals[0], self.evals[-1])

        self.eval_half = np.zeros(len(self.evals)-1)
        for i in range(len(self.eval_half)):
            self.eval_half[i] = (self.evals[i]+self.evals[i+1])/2

        self.mX = np.atleast_1d(mX)
        self.N_mX = len(self.mX)
        self.mX_str = []
        for mm in self.mX:
            self.mX_str.append(r'$m_{}={}$ MeV'.format('{\chi}', int(mm*1e-6)))

        self.fdmn = np.atleast_1d(fdmn)
        self.N_fdmn = len(self.fdmn)
        self.fdm_str = [r'$F_{DM}=1$', r'$F_{DM} \sim 1/q^2$']

        self.isLeg = False
        self.isRect = False

        self.gs = gridspec.GridSpec(ncols=self.N_fdmn, nrows=self.N_mX,
                                    figure=fig_)
        self.gs.update(wspace=0.0, hspace=0.0)

        self.axx = []

        for i in range(self.N_mX):
            axtmp = []
            for j in range(self.N_fdmn):
                tmp1 = fig_.add_subplot(self.gs[i,j], xlim=xlims)
                axtmp.append(tmp1)
            self.axx.append(axtmp)

        for ll in self.axx:
            for ax_ in ll:
                ax_.set_yscale('log')

        for i in range(self.N_mX):
            self.axx[i][0].set_ylabel(r'Rate [\# / kg-year]')

            if self.N_fdmn > 1:
                for j in range(1, self.N_fdmn):
                    self.axx[i][j].yaxis.tick_right()

            for j in range(self.N_fdmn):
                self.axx[i][j].text(0.05, 0.05, self.fdm_str[j],
                                    transform=self.axx[i][j].transAxes,
                                    fontsize=12, verticalalignment='bottom')
                self.axx[i][j].text(0.05, 0.12, self.mX_str[i],
                                    transform=self.axx[i][j].transAxes,
                                    fontsize=12, verticalalignment='bottom')

        for ax_ in self.axx[0]:
            ax_.set_ylim(self.ylim[0])
            ax_.set_yticks(self.ytck[0])
            if self.if_ylab[0]:
                ax_.set_yticklabels(self.ylab[0])
            ax_.set_xticks(self.evals[:-1])
            ax_.set_xticklabels(np.around(self.evals[:-1], decimals=1),
                                fontsize=10)
            ax_.xaxis.tick_top()
            ax_.tick_params(which='major', direction='in', bottom=True,
                            right=True, left=True)
            ax_.set_xlabel(r'$E_e$ [eV]', fontsize=10)
            ax_.xaxis.set_label_position('top')

        for ax_ in self.axx[-1]:
            ax_.set_ylim(self.ylim[-1])
            ax_.set_yticks(self.ytck[-1])
            if self.if_ylab[-1]:
                ax_.set_yticklabels(self.ylab[-1])
            ax_.set_xticks(self.evals)
            ax_.set_xticklabels([])
            ax_.set_xticks(self.eval_half, minor=True)
            ax_.set_xticklabels(np.arange(1, len(self.evals)), minor=True)
            ax_.tick_params(which='minor', length=0.0)
            ax_.tick_params(which='major', direction='in', right=True, top=True,
                            left=True)
            ax_.set_xlabel(r'$N_e$')

        plt.subplots_adjust(bottom=.18)
        # plt.subplots_adjust(top=.5)

        self.legs = []
        self.lines = []
        self.rects = []

        if plot_shm:
            self.fidvals = np.zeros( (self.N_mX, self.N_fdmn, self.N_bins) )
            db_path = CrysObj.dpath
            tmpCrys = Crystal_DMe('Si', db_path)
            for j in range(self.N_fdmn):
                tmpCrys.set_params( FDMn = self.fdmn[j] )
                for i in range(self.N_mX):
                    self.fidvals[i,j,:] = tmpCrys.Rate( self.mX[i],
                                                        Ne_max=self.N_bins )
                    self.axx[i][j].hist(self.evals[:-1], self.evals,
                                        weights=self.fidvals[i,j], color='k',
                                        lw=1., histtype='step')
            self.lines.append( Line2D([], [], c='k', lw=1.,
                               label='SHM fiducial') )

        else:
            if not 'label' in kwargs:
                label = CrysObj.vname
            else:
                label = kwargs['label']
            self.fidvals = np.zeros( (self.N_mX, self.N_fdmn, self.N_bins) )
            for j in range(self.N_fdmn):
                CrysObj.set_params( FDMn = self.fdmn[j] )
                for i in range(self.N_mX):
                    self.fidvals[i,j,:] = CrysObj.Rate( self.mX[i],
                                                        Ne_max=self.N_bins )
                    self.axx[i][j].hist(self.evals[:-1], self.evals,
                                        weights=self.fidvals[i,j], color='k',
                                        lw=1., histtype='step')
            self.lines.append(Line2D([], [], c='k', lw=1., label=label))

        self.set_legend()

        fig_.suptitle( Title, y=.925 )

    def set_params(self, **kwargs):
        allowed_keys = {'ylim', 'ytck', 'if_ylab', 'ylab'}
        self.__dict__.update((k,v) for k,v in kwargs.items() if k in allowed_keys)

    def set_legend(self):

        if self.isLeg:
            self.legend.remove()

        legs = []

        if self.isRect:
            linetmp = []
            recttmp = []
            for ll in self.lines:
                linetmp.append(ll)
            for rr in self.rects:
                recttmp.append(rr)
            if len(linetmp) != len(recttmp):
                while len(linetmp) < len(recttmp):
                    linetmp.append(Line2D([0], [0], color="w", label=''))
                while len(recttmp) < len(linetmp):
                    recttmp.append(Line2D([0], [0], color="w", label=''))

            for i in range(len(linetmp)):
                legs.append(linetmp[i])
                legs.append(recttmp[i])

            self.legend = plt.legend(handles=legs, loc='lower center',
                                     bbox_to_anchor=(.5,0),
                                     bbox_transform=self.fig.transFigure,
                                     ncol=len(linetmp), fontsize=12)

        else:
            for ll in self.lines:
                legs.append(ll)
            self.legend = plt.legend(handles=legs, loc='lower center',
                                     bbox_to_anchor=(.5,0),
                                     bbox_transform=self.fig.transFigure,
                                     ncol=int(np.ceil(len(self.lines)/2)),
                                     fontsize=12)

        self.isLeg = True

    def add_line(self, CrysObj, label=None, c=None, ls='solid', **kwargs):
        Vals = np.zeros_like(self.fidvals)

        if not 'lw' in kwargs:
            lw=1.
        else:
            lw=kwargs['lw']
            del kwargs['lw']

        for j in range(self.N_fdmn):
            CrysObj.set_params( FDMn = self.fdmn[j] )

            for i in range(self.N_mX):
                Vals[i,j,:] = CrysObj.Rate( self.mX[i], Ne_max=self.N_bins )
                self.axx[i][j].hist(self.evals[:-1], self.evals,
                               weights=Vals[i,j], color=c, lw=lw, linestyle=ls,
                               histtype='step', **kwargs)
        self.lines.append( Line2D([], [], color=c, ls=ls, lw=lw,
                           label=label, **kwargs) )

        self.set_legend()

    def add_band(self, CrysObj_low, CrysObj_high, label=None, c=None,
                 hatch=None):
        Vals_low  = np.zeros_like(self.fidvals)
        Vals_high = np.zeros_like(self.fidvals)

        for j in range(self.N_fdmn):
            CrysObj_low.set_params( FDMn = self.fdmn[j] )
            CrysObj_high.set_params( FDMn = self.fdmn[j] )

            for i in range(self.N_mX):
                Vals_low[i,j,:] = CrysObj_low.Rate( self.mX[i],
                                                    Ne_max=self.N_bins )
                Vals_high[i,j,:] = CrysObj_high.Rate( self.mX[i],
                                                      Ne_max=self.N_bins )
                self.axx[i][j].hist(self.evals[:-1], self.evals,
                                    weights=(Vals_high[i,j] - Vals_low[i,j]),
                                    bottom=Vals_low[i,j], color=c, hatch=hatch,
                                    alpha=0.3, ec=c)
        self.rects.append( Rectangle((0,0), 1, 1, color=c,
                           hatch=hatch, alpha=0.3, ec=c,
                           label=label) )
        self.isRect = True

        self.set_legend()


"""                           Cross-section Class                           """


class Sigma_figure:

    def __init__(self, fig_, Title, CrysObj, exposure, mX, Ne=[1,2,3],
                 fdmn=[0,2], plot_shm=True, cband=True, **kwargs):

        params = {'text.usetex' : True,
                  'font.size' : 14,
                  'font.family' : 'cmr10',
                  'figure.autolayout': True}
        plt.rcParams.update(params)
        plt.rcParams['axes.unicode_minus']=False
        plt.rcParams['axes.labelsize']=16

        self.fig = fig_
        self.fig.set_size_inches(10,5)

        self.isLeg = False

        self.ylims = (.5e-42, 2e-38)
        self.ytcks = [1e-42, 1e-41, 1e-40, 1e-39, 1e-38]

        self.exposure = exposure

        self.mX = np.atleast_1d(mX)
        self.mX_MeV = self.mX*1e-6
        self.N_mX = len(self.mX)
        self.xlims = (self.mX_MeV[0], self.mX_MeV[-1])
        self.xtcks = [1e0, 1e1, 1e2]

        self.Ne = Ne
        self.N_Ne = len(Ne)
        self.Nelabs = [r'$N_e=$' + str(ne) for ne in Ne]
        self.Ne_colors = ['purple', 'tab:orange', 'tab:green']

        self.fdmn = fdmn
        self.N_fdmn = len(fdmn)
        self.fdm_str = [r'$F_{DM}=1$', r'$F_{DM} \sim 1/q^2$']

        self.set_params(**kwargs)

        self.gs = gridspec.GridSpec(ncols=4, nrows=2, figure=fig_)
        self.gs.update(wspace=0.0)
        self.ax1 = fig_.add_subplot(self.gs[:, :2], xlim=self.xlims,
                                    ylim=self.ylims)
        self.ax2 = fig_.add_subplot(self.gs[:, 2:], xlim=self.xlims,
                                    ylim=self.ylims)

        for ax_ in fig_.axes:
            ax_.set_xscale('log')
            ax_.set_yscale('log')
            ax_.set_xticks(self.xtcks)
            ax_.set_yticks(self.ytcks)
            ax_.tick_params(axis='y', direction='in', which='both')
        self.ax2.yaxis.tick_right()
        self.ax1.tick_params(right=True, top=True, direction='in', which='both')
        self.ax2.tick_params( left=True, top=True, direction='in', which='both')

        plt.subplots_adjust(bottom=.25)
        # plt.subplots_adjust(top=.3)

        self.lines = []
        self.NeLines = []

        if plot_shm:
            db_path = CrysObj.dpath
            tmpCrys = Crystal_DMe('Si', db_path)
            self.fidvals = np.zeros( (self.N_fdmn, self.N_Ne, self.N_mX) )
            self.lines.append(Line2D([], [], color='k', label='SHM fiducial'))
            for i in range(self.N_fdmn):
                tmpCrys.set_params( FDMn=self.fdmn[i] )
                for j in range(self.N_Ne):
                    self.fidvals[i,j,:] = tmpCrys.sig_min(self.mX,
                                                          self.exposure,
                                                          self.Ne[j])
                    for m in range(self.N_mX):
                        if self.fidvals[i,j,m] > 1e36:
                            self.fidvals[i,j,m] = 1e36
                    tmp1, = self.fig.axes[i].plot( self.mX_MeV,
                                                   self.fidvals[i,j],
                                                   color=self.Ne_colors[j],
                                                   label=self.Nelabs[j] )
                    if i == 0:
                        self.NeLines.append(tmp1)
            if cband:
                from utils.astroparams_c import v0min, vEmin, vescmin, v0max, \
                                                vEmax, vescmax
                tmpC_low  = Crystal_DMe('Si', db_path,
                                        vparams=[v0min, vEmin, vescmin])
                tmpC_high = Crystal_DMe('Si', db_path,
                                        vparams=[v0max, vEmax, vescmax])
                self.add_bands(tmpC_low, tmpC_high)

        else:
            if not 'label' in kwargs:
                label = CrysObj.vname
            else:
                label = kwargs['label']
            self.fidvals = np.zeros( (self.N_fdmn, self.N_Ne, self.N_mX) )
            self.lines.append(Line2D([], [], color='k', label=label))
            for i in range(self.N_fdmn):
                CrysObj.set_params( FDMn=self.fdmn[i] )
                for j in range(self.N_Ne):
                    self.fidvals[i,j,:] = CrysObj.sig_min(self.mX,
                                                          self.exposure,
                                                          self.Ne[j])
                    for m in range(self.N_mX):
                        if self.fidvals[i,j,m] > 1e36:
                            self.fidvals[i,j,m] = 1e36
                    tmp1, = self.fig.axes[i].plot( self.mX_MeV,
                                                   self.fidvals[i,j],
                                                   color=self.Ne_colors[j],
                                                   label=self.Nelabs[j] )
                    if i == 0:
                        self.NeLines.append(tmp1)

        self.set_legend()

        self.ax1.set_xlabel(r'$m_{\chi}$ [MeV]')
        self.ax1.set_ylabel(r'$\bar{\sigma}_e$ [cm$^2$]')
        self.ax2.set_xlabel(r'$m_{\chi}$ [MeV]')
        self.ax1.text(0.7, 0.05, r'$F_{DM}=1$', transform=self.ax1.transAxes,
                      fontsize=12, verticalalignment='bottom')
        self.ax2.text(0.7, 0.05, r'$F_{DM} \sim 1/q^2$',
                      transform=self.ax2.transAxes, fontsize=12,
                      verticalalignment='bottom')

        self.fig.suptitle( Title, y=.925 )

    def set_params(self, **kwargs):
        allowed_keys = {'ylims', 'ytcks', 'xtcks', 'fdm_str'}
        self.__dict__.update((k,v) for k,v in kwargs.items() if k in allowed_keys)

    def set_legend(self):

        if self.isLeg:
            self.legend.remove()

        legs = []

        linetmp = []
        Netmp = []
        for ll in self.lines:
            linetmp.append(ll)
        for nn in self.NeLines:
            Netmp.append(nn)
        if len(linetmp) != len(Netmp):
            while len(linetmp) < len(Netmp):
                linetmp.append(Line2D([0], [0], color="w", label=''))
            while len(Netmp) < len(linetmp):
                Netmp.append(Line2D([0], [0], color="w", label=''))

        for i in range(len(linetmp)):
            legs.append(linetmp[i])
            legs.append(Netmp[i])

        self.legend = plt.legend(handles=legs, loc='lower center',
                                 bbox_to_anchor=(.5,0),
                                 bbox_transform=self.fig.transFigure,
                                 ncol=len(linetmp), fontsize=12)

        self.isLeg = True

    def add_line(self, CrysObj, label=None, ls='solid', **kwargs):
        if 'mX' in kwargs:
            mX = np.atleast_1d(kwargs['mX'])
            mX_MeV = 1e-6*mX
            N_mX = len(mX)
            del kwargs['mX']
        else:
            mX = self.mX
            mX_MeV = self.mX_MeV
            N_mX = self.N_mX
        vals = np.zeros( (self.N_fdmn, self.N_Ne, N_mX) )
        for i in range(self.N_fdmn):
            CrysObj.set_params( FDMn=self.fdmn[i] )
            for j in range(self.N_Ne):
                vals[i,j,:] = CrysObj.sig_min(mX, self.exposure, self.Ne[j])
                for m in range(N_mX):
                    if vals[i,j,m] > 1e36:
                        vals[i,j,m] = 1e36
                self.fig.axes[i].plot( mX_MeV, vals[i,j],
                                       color=self.Ne_colors[j], ls=ls,
                                       **kwargs )
        self.lines.append(Line2D([], [], color='k', ls=ls, label=label))
        self.set_legend()

    def add_bands(self, CrysObj_low, CrysObj_high, **kwargs):
        vals = np.zeros( (2, self.N_fdmn, self.N_Ne, self.N_mX) )
        for i in range(self.N_fdmn):
            CrysObj_low.set_params( FDMn=self.fdmn[i] )
            CrysObj_high.set_params( FDMn=self.fdmn[i] )
            for j in range(self.N_Ne):
                vals[0,i,j,:] = CrysObj_low.sig_min(self.mX, self.exposure,
                                                    self.Ne[j])
                vals[1,i,j,:] = CrysObj_high.sig_min(self.mX, self.exposure,
                                                     self.Ne[j])
                for m in range(self.N_mX):
                    for n in range(2):
                        if vals[n,i,j,m] > 1e36:
                            vals[n,i,j,m] = 1e36
                self.fig.axes[i].fill_between(self.mX_MeV, vals[0,i,j],
                                              vals[1,i,j],
                                              color=self.Ne_colors[j],
                                              alpha=0.3, **kwargs)
