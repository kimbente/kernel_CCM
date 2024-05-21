import plotly.graph_objects as go
import torch

#################
### Synchrony ###
#################

def vis_synchrony(shift_results_gpccm, k, shifts, true_shift, isGPCCM, legend_y_displacement = 1.01):

    if (isGPCCM == True):
        line_col = "#C00000"
        bounds_line_col = 'rgba(243, 176, 210, 0.8)'
        bounds_fill_col = 'rgba(243, 176, 210, 0.3)'

    else:
        line_col = "rgba(103, 95, 114, 1.0)"
        bounds_line_col = "rgba(134, 122, 151, 0.8)"
        bounds_fill_col = "rgba(134, 122, 151, 0.3)"

    
    # Initalise
    fig = go.Figure()

    # left to right
    fig.add_vrect(x0 = -8, x1 = -(k - 1), line_width = 0, fillcolor = "green", opacity = 0.1)
    fig.add_vrect(x0 = -(k - 1), x1 = 0.0, line_width = 0, fillcolor = "grey", opacity = 0.2)
    fig.add_vrect(x0 = 0., x1 = 8., line_width = 0, fillcolor = "red", opacity = 0.1)

    # separators
    fig.add_vline(x = -(k - 1), line_width = 0.5, line_color = "black", opacity = 0.5)
    fig.add_vline(x = 0.0, line_width = 0.5, line_color = "black", opacity = 0.5)

    # fig.add_vline(x = true_shift - 1, line_width = 1.0, line_color = "green", line_dash = "dash", opacity = 0.8)

    fig.add_trace(go.Scatter(x = -shifts, y = shift_results_gpccm[:, 2], # reversing the meaning of x
                        mode = 'lines',
                        name = 'rho ind. p95',
                        line_color = "black",
                        line_dash = "dot",
                        ))

    fig.add_trace(go.Scatter(x = -shifts, y = shift_results_gpccm[:, 0], # reversing the meaning of x
                        mode = 'lines+markers',
                        name = 'mean rho +/- 1 sd',
                        line_color = line_col))


    fig.add_trace(go.Scatter(
        name = "",
        x = - shifts,
        y = shift_results_gpccm[:, 0] + shift_results_gpccm[:, 1].mul(1),
        marker = dict(color = bounds_line_col),
        showlegend = False,
        mode = 'lines'
    ))

    fig.add_trace(go.Scatter(
        name = "",
        x = - shifts,
        y = shift_results_gpccm[:, 0] - shift_results_gpccm[:, 1].mul(1),
        marker = dict(color = bounds_line_col),
        showlegend = False,
        mode = 'lines',
        fillcolor = bounds_fill_col,
        fill = 'tonexty'
    ))

    # empty
    fig.update_layout(title = '',
                    xaxis_title = '',
                    yaxis_title = '')

    fig.update_layout(template = "simple_white")
    fig.update_layout(font_family = "Lato")
    fig.update_layout(width = 600, height = 350)
    fig.update_layout(xaxis_range = [-8.0, 8.0])
    fig.update_layout(yaxis_range = [-0.1, 1.01])

    fig.update_layout(legend = dict(x = 0, y = legend_y_displacement, bgcolor = "rgba(0,0,0,0)"))

    fig.show()