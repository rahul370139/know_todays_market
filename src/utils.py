import pandas as pd
import pickle
import plotly.graph_objs as go


def load_dataframe(path: str) -> pd.DataFrame:
    """Read CSV into a DataFrame."""
    return pd.read_csv(path)


def save_dataframe(df: pd.DataFrame, path: str):
    """Write DataFrame to CSV."""
    df.to_csv(path, index=False)


def save_pickle(obj, path: str):
    """Serialize Python object to disk."""
    with open(path, 'wb') as f:
        pickle.dump(obj, f)


def load_pickle(path: str):
    """Load pickled Python object."""
    with open(path, 'rb') as f:
        return pickle.load(f)


def plot_actual_vs_predicted(dates, actual, predicted, target: str):
    """
    Line chart of actual vs. predicted.
    """
    fig = go.Figure()
    fig.add_trace(go.Scatter(
        x=dates, y=actual,
        mode='lines+markers',
        name='Actual'
    ))
    fig.add_trace(go.Scatter(
        x=dates, y=predicted,
        mode='lines+markers',
        name='Predicted'
    ))
    fig.update_layout(
        title=f"{target} — Actual vs Predicted",
        xaxis_title="Date",
        yaxis_title=target,
        template='plotly_white',
        hovermode='x unified'
    )
    fig.show()


def plot_bar_predictions(pred_df: pd.DataFrame,
                         title: str,
                         xcol: str = 'Target',
                         ycol: str = 'Predicted'):
    """
    Bar chart for next‐day predictions.
    """
    fig = go.Figure(go.Bar(
        x=pred_df[xcol],
        y=pred_df[ycol]
    ))
    fig.update_layout(
        title=title,
        xaxis_title=xcol,
        yaxis_title=ycol,
        template='plotly_white'
    )
    fig.show()
