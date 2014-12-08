import copy
import operator
import husl

import numpy as np

import matplotlib.pyplot as plt

from stats import compute_R2


def multi_plot(the_data_list, plot_func, title=None, nrows=4, ncols=5, figsize=None, output_pattern=None, transpose=False, facecolor='gray'):

    nsp = 0
    fig = None
    fig_num = 0
    plots_per_page = nrows*ncols

    data_list = the_data_list
    overflow_index = 0
    if transpose:
        data_list = [None]*len(data_list)
        for k in range(len(the_data_list)):
            page_offset = int(float(k) / plots_per_page)*plots_per_page
            if len(the_data_list) - page_offset < plots_per_page:
                new_index = page_offset + overflow_index
                overflow_index += 1
            else:
                sp = k % plots_per_page
                row = sp % nrows
                col = int(float(sp) / nrows)
                new_index = page_offset + row*ncols + col
            print 'nsp=%d, k=%d, sp=%d, page_offset=%d, row=%d, col=%d, new_index=%d' % \
                  (len(the_data_list), k, sp, page_offset, row, col, new_index)
            data_list[new_index] = the_data_list[k]

    for pdata in data_list:
        if nsp % plots_per_page == 0:
            if output_pattern is not None and fig is not None:
                #save the current figure
                ofile = output_pattern % fig_num
                plt.savefig(ofile, facecolor=fig.get_facecolor(), edgecolor='none')
                plt.close('all')
            fig = plt.figure(figsize=figsize, facecolor=facecolor)
            fig_num += 1
            fig.subplots_adjust(top=0.95, bottom=0.02, right=0.97, left=0.03, hspace=0.20)
            if title is not None:
                plt.suptitle(title + (" (%d)" % fig_num))

        nsp += 1
        sp = nsp % plots_per_page
        ax = fig.add_subplot(nrows, ncols, sp)
        plot_func(pdata, ax)

    #save last figure
    if fig is not None and output_pattern is not None:
        ofile = output_pattern % fig_num
        plt.savefig(ofile, facecolor=fig.get_facecolor(), edgecolor='none')
        plt.close('all')


def plot_pairwise_analysis(data_mat, feature_columns, dependent_column, column_names):
    """
        Does a basic pairwise correlation analysis between features and a dependent variable,
        meaning it plots a scatter plot with a linear curve fit through it, with the R^2.
        Then it plots a correlation matrix for all features and the dependent variable.

        data_mat: an NxM matrix, where there are N samples, M-1 features, and 1 dependent variable.

        feature_columns: the column indices of the features in data_mat that are being examined

        dependent_column: the column index of the dependent variable in data_mat

        column_names: a list of len(feature_columns)+1 feature/variable names. The last element is
                      the name of the dependent variable.
    """

    plot_data = list()
    for k,fname in enumerate(column_names[:-1]):
        fi = feature_columns[k]

        pdata = dict()
        pdata['x'] = data_mat[:, fi]
        pdata['y'] = data_mat[:, dependent_column]
        pdata['xlabel'] = column_names[fi]
        pdata['ylabel'] = column_names[-1]
        pdata['R2'] = compute_R2(pdata['x'], pdata['y'])
        plot_data.append(pdata)

    #sort by R^2
    plot_data.sort(key=operator.itemgetter('R2'), reverse=True)
    multi_plot(plot_data, plot_pairwise_scatter, title=None, nrows=3, ncols=3)

    all_columns = copy.copy(feature_columns)
    all_columns.append(dependent_column)

    C = np.corrcoef(data_mat[:, all_columns].transpose())

    Cy = C[:, -1]
    corr_list = [(column_names[k], np.abs(Cy[k]), Cy[k]) for k in range(len(column_names)-1)]
    corr_list.sort(key=operator.itemgetter(1), reverse=True)

    print 'Correlations  with %s' % column_names[-1]
    for cname,abscorr,corr in corr_list:
        print '\t%s: %0.6f' % (cname, corr)

    fig = plt.figure()
    plt.subplots_adjust(top=0.99, bottom=0.15, left=0.15)
    ax = fig.add_subplot(1, 1, 1)
    fig.autofmt_xdate(rotation=45)
    im = ax.imshow(C, interpolation='nearest', aspect='auto', vmin=-1.0, vmax=1.0, origin='lower')
    plt.colorbar(im)
    ax.set_yticks(range(len(column_names)))
    ax.set_yticklabels(column_names)
    ax.set_xticks(range(len(column_names)))
    ax.set_xticklabels(column_names)


def plot_pairwise_scatter(plot_data, ax):

    x = plot_data['x']
    y = plot_data['y']
    if 'R2' not in plot_data:
        R2 = compute_R2(x, y)
    else:
        R2 = plot_data['R2']
    slope,bias = np.polyfit(x, y, 1)
    sp = (x.max() - x.min()) / 25.0
    xrng = np.arange(x.min(), x.max(), sp)

    clr = '#aaaaaa'
    if 'color' in plot_data:
        clr = plot_data['color']
    ax.plot(x, y, 'o', mfc=clr)
    ax.plot(xrng, slope*xrng + bias, 'k-')
    ax.set_title('%s: R2=%0.2f' % (plot_data['xlabel'], R2))
    if 'ylabel' in plot_data:
        ax.set_ylabel(plot_data['ylabel'])
    ax.set_ylim(y.min(), y.max())


def plot_histogram_categorical(x, xname='x', y=None, yname='y', color='g'):
    """
        Makes a histogram of the variable x, which is an array of categorical variables in their native representation
        (string or intger) . If a dependent continuous variable y is specified, it will make another plot which
        is a bar graph showing the mean and standard deviation of the continuous variable y.
    """

    ux = np.unique(x)
    xfracs = np.array([(x == xval).sum() for xval in ux]) / float(len(x))

    nsp = 1 + (y is not None)
    ind = range(len(ux))

    fig = plt.figure()
    ax = fig.add_subplot(nsp, 1, 1)
    ax.bar(ind, xfracs, facecolor=color, align='center', ecolor='black')
    ax.set_xticks(ind)
    ax.set_xticklabels(ux)
    ax.set_xlabel(xname)
    ax.set_ylabel('Fraction of Samples')

    if y is not None:
        y_per_x = dict()
        for xval in ux:
            indx = x == xval
            y_per_x[xval] = y[indx]

        ystats = [ (xval, y_per_x[xval].mean(), y_per_x[xval].std()) for xval in ux]
        ystats.sort(key=operator.itemgetter(0), reverse=True)

        xvals = [x[0] for x in ystats]
        ymeans = np.array([x[1] for x in ystats])
        ystds = np.array([x[2] for x in ystats])

        ax = fig.add_subplot(nsp, 1, 2)
        ax.bar(ind, ymeans, yerr=ystds, facecolor=color, align='center', ecolor='black')
        ax.set_xticks(ind)
        ax.set_xticklabels(xvals)
        #fig.autofmt_xdate()
        ax.set_ylabel('Mean %s' % yname)
        ax.set_xlabel(xname)
        ax.set_ylim(0, (ymeans+ystds).max())


def whist(x, **kwds):
    return plt.hist(x, weights=np.ones([len(x)]) / float(len(x)), **kwds)


def plot_confusion_matrix_single(pdata, ax):
    plt.imshow(pdata['cmat'], interpolation='nearest', aspect='auto', origin='upper', vmin=0, vmax=1)
    plt.title('p=%0.3f' % pdata['p'])


def make_phase_image(amp, phase, normalize=True, saturate=True, threshold=True):
    """
        Turns a phase matrix into an image to be plotted with imshow.
    """

    nelectrodes,d = amp.shape
    alpha = copy.deepcopy(amp)
    if normalize:
        max_amp = np.percentile(amp, 98)
        alpha = alpha / max_amp

    img = np.zeros([nelectrodes, d, 4], dtype='float32')

    #set the alpha and color for the bins
    if saturate:
        alpha[alpha > 1.0] = 1.0 #saturate
    if threshold:
        alpha[alpha < 0.05] = 0.0 #nonlinear threshold

    cnorm = ((180.0 / np.pi) * phase).astype('int')
    for j in range(nelectrodes):
        for ti in range(d):
            #img[j, ti, :3] = husl.husl_to_rgb(cnorm[j, ti], 99.0, 50.0) #use HUSL color space: https://github.com/boronine/pyhusl/tree/v2.1.0
            img[j, ti, :3] = husl.husl_to_rgb(cnorm[j, ti], 99.0, 61.0) #use HUSL color space: https://github.com/boronine/pyhusl/tree/v2.1.0

    img[:, :, 3] = alpha

    return img


def draw_husl_circle():
    """ Draw an awesome circle whose angle is colored using the HUSL color space. The HUSL color space is circular, so
        it's useful for plotting phase. This figure could serve as a "color circle" as opposed to a color bar.
    """

    #generate a bunch of points on the circle
    theta = np.arange(0.0, 2*np.pi, 1e-3)

    plt.figure()

    radii = np.arange(0.75, 1.0, 1e-2)
    for t in theta:
        x = radii*np.cos(t)
        y = radii*np.sin(t)

        a = (180.0/np.pi)*t
        c = husl.husl_to_rgb(a, 99.0, 50.0)
        plt.plot(x, y, c=c)
    plt.xlim(-1.25, 1.25)
    plt.ylim(-1.25, 1.25)

    plt.show()

if __name__ == '__main__':

    draw_husl_circle()

