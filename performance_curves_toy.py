import plotly
from matplotlib import  pyplot as plt
plotly.io.orca.config.executable = '/home/hx/.miniconda3/envs/combi/bin/orca'
import plotly.graph_objects as go
import numpy as np

"""
global_layout = dict(shapes=[

    # Line Horizontal
    go.layout.Shape(
        type="line",
        x0=2,
        y0=3,
        x1=10020,
        y1=3,
        name='t_star',
        line=dict(
            color="Red",
            width=4
        ),
    )
],
    title=go.layout.Title(
        text="P-Values distribution",
        xref="paper",
        x=0
    ),
    legend=go.layout.Legend(
        x=0,
        y=1
    ),
    xaxis=go.layout.XAxis(
        title=go.layout.xaxis.Title(
            text="Locus",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    ),
    yaxis=go.layout.YAxis(
        title=go.layout.yaxis.Title(
            text="-log(pvalue)",
            font=dict(
                family="Courier New, monospace",
                size=18,
                color="#7f7f7f"
            )
        )
    )
)

array = np.random.uniform(low=0, high=1, size=10020)
info = np.random.uniform(low=0, high=5, size=20)
idx = np.random.choice(10020, 20)

pts = zip(idx, info)
pts_greater = np.array([pt for pt in pts if pt[1] > 3])

pts = zip(idx, info)
pts_lower = np.array([pt for pt in pts if 1.1 <= pt[1] <= 3])

fig = go.Figure(data=go.Scatter(x=np.arange(len(array)),
                                y=array,
                                marker_color="#104BA9",
                                opacity=0.5,
                                name='Uninformative SNPs',
                                mode='markers'))

fig.add_trace(go.Scatter(x=pts_greater[:, 0],
                         y=pts_greater[:, 1],
                         marker_color="#D9005B",
                         opacity=1.0,
                         name='Informative SNPs',
                         mode='markers'))

fig.add_trace(go.Scatter(x=pts_lower[:, 0],
                         y=pts_lower[:, 1],
                         opacity=0.5,
                         marker_color="#104BA9",

                         showlegend=False,
                         mode='markers'))

fig.update_layout(
    global_layout,
    title="Using P-Values to isolate informative SNPs")

fig.show()
# fig.write_image("/home/hx/Work/Masterarbeit/report/slides/pvalues.pdf")


import plotly

plotly.io.orca.config.executable = '/home/hx/.miniconda3/envs/combi/bin/orca'
import plotly.graph_objects as go
import numpy as np

fig2 = go.Figure(data=go.Scatter(x=np.arange(5000, 5020),
                                 y=np.random.uniform(low=3, high=5, size=20),
                                 marker_color="#D9005B",
                                 opacity=1,
                                 name='Informative SNPs',
                                 mode='markers'))

fig2.add_trace(go.Scatter(x=np.array([np.arange(5000), np.arange(5020, 10020)]).flatten(),
                          y=np.random.uniform(low=0, high=1, size=10000),
                          marker_color="#104BA9",
                          opacity=0.5,
                          name='Uninformative SNPs',
                          mode='markers'))

fig2.update_layout(
    global_layout,
    shapes=[

        go.layout.Shape(
            type="rect",
            x0=4080,
            y0=0,
            x1=5000,
            y1=5,
            opacity=0.2,
            layer='below',
            line_width=0,
            fillcolor="#104BA9",
        ),
        # filled Rectangle
        go.layout.Shape(
            type="rect",
            x0=5000,
            y0=0,
            x1=5020,
            y1=5,
            opacity=0.2,
            layer='below',
            line_width=0,
            fillcolor="#D9005B",
        ),

        go.layout.Shape(
            type="rect",
            x0=5020,
            y0=0,
            x1=5040,
            y1=5,
            opacity=0.2,
            layer='below',
            line_width=0,
            fillcolor="#104BA9",
        ),
    ]
)
fig2.update_xaxes(range=[4980, 5040])
#fig2.show()
#fig2.write_image("/home/hx/Work/Masterarbeit/report/slides/toy.pdf")
"""

colors = ["#E59F71", "#BA5A31", "#97A877", "#69DC9E", "#FF9B9B", "#306448", "#552917", "#6E7B57"]
tpr = {}
fwer = {}
precision = {}
tpr['31'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/tpr-31-6.npy")
tpr['35'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/tpr-35-6.npy")
tpr['41'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/tpr-41-6.npy")
tpr['rpvt'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/rpvt-tpr-6.npy")
tpr['combi'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/combi-tpr-6.npy")

precision['31'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/precision-31-6.npy")
precision['35'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/precision-35-6.npy")
precision['41'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/precision-41-6.npy")
precision['rpvt'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/rpvt-precision-6.npy")
precision['combi'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/combi-precision-6.npy")

fwer['31'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/fwer-31-6.npy")
fwer['35'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/fwer-35-6.npy")
fwer['41'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/fwer-41-6.npy")
fwer['rpvt'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/rpvt-fwer-6.npy")
fwer['combi'] = np.load("/home/hx/Work/Masterarbeit/numpy_arrays_2/combi-fwer-6.npy")



fig, ax = plt.subplots(1,1)

ax.plot(tpr['31'], precision['31'],'x-', tpr['rpvt'], precision['rpvt'],'x-', tpr['combi'],
        precision['combi'],'x-')

plt.legend(loc='lower left', labels=['DeepCOMBI', 'RPVT', 'COMBI'])
plt.xlabel('True Positive Rate')
plt.ylabel('Precision')
plt.tight_layout()

ax.figure.savefig("/home/hx/Work/Masterarbeit/report/report/lrp-combi-tpr-fwer.png")

# TPR/FWER
fig, ax = plt.subplots(1,1)
ax.plot(fwer['31'], tpr['31'],'x-', fwer['rpvt'], tpr['rpvt'],'x-', fwer['combi'],
        tpr['combi'],'x-')
ax.set_xlim(0, 0.1)
ax.set_ylim(0, 0.55)

plt.legend(loc='lower left', labels=['DeepCOMBI', 'RPVT', 'COMBI'])
plt.xlabel('Family-wise Error Rate')
plt.ylabel('True Positive Rate')
plt.tight_layout()

ax.figure.savefig("/home/hx/Work/Masterarbeit/report/report/lrp-combi-precision-tpr.png")
