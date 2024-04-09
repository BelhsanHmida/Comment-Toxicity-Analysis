import streamlit as st
import pandas as pd
import plotly.graph_objects as go

def create_pie_plot(data, category):
    toxic_color = '#ff0000'      # Bright red
    non_toxic_color = '#00ff00'  # Bright green

    fig = go.Figure(go.Pie(
        labels=['Non-toxic', 'Toxic'],
        values=[len(data[data[category] == 0]), len(data[data[category] == 1])],
        hole=0.6,
        marker=dict(colors=[toxic_color, non_toxic_color])
    ))

    fig.update_layout(
        title=f'Distribution of {category} Comments',
        margin=dict(l=10, r=10, t=50, b=50),
        paper_bgcolor='rgba(0,0,0,0)',
        plot_bgcolor='rgba(0,0,0,0)'
    )

    return fig

def main():
    st.title('Toxic Comments Analysis')

    # Sample DataFrame
    data = {
        'comment_text': ['This is a toxic comment', 'This is not toxic'],
        'toxic': [1, 0],
        'severe_toxic': [0, 0],
        'obscene': [1, 0],
        'threat': [0, 0],
        'insult': [1, 0],
        'identity_hate': [0, 0]
    }
    df = pd.DataFrame(data)

    st.header('Distribution of Toxic Comments')
    for category in df.columns[1:]:
        st.plotly_chart(create_pie_plot(df, category))

if __name__ == "__main__":
    main()
