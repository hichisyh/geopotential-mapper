import plotly.graph_objects as go


def anomaly_map(work, x_col, y_col, value_col, xx, yy, zz, title, colorscale='Turbo'):
    fig = go.Figure()
    fig.add_trace(go.Contour(
        x=xx[0, :],
        y=yy[:, 0],
        z=zz,
        colorscale=colorscale,
        contours=dict(showlabels=True),
        colorbar=dict(title=value_col),
        name='Grid',
    ))
    fig.add_trace(go.Scatter(
        x=work[x_col],
        y=work[y_col],
        mode='markers',
        marker=dict(size=5, color='black'),
        name='Stations',
        text=[f'{v:.3f}' for v in work[value_col]],
        hovertemplate='X=%{x}<br>Y=%{y}<br>Value=%{text}<extra></extra>',
    ))
    fig.update_layout(
        title=title,
        xaxis_title=x_col,
        yaxis_title=y_col,
        height=720,
        margin=dict(l=40, r=40, t=70, b=40),
    )
    fig.update_yaxes(scaleanchor='x', scaleratio=1)
    return fig
