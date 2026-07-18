import plotly.express as px
import plotly.graph_objects as go
import streamlit as st
from streamlit_autorefresh import st_autorefresh
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt


model = joblib.load("SmartSense_AI.pkl")

# 👇 ADD THE SIDEBAR HERE
st.sidebar.title("Navigation")
st.sidebar.success("AI Wi-Fi Detection")
st.sidebar.info("""Developer:
Khush Prajapati 

B.Tech CSE AIML 

Machine Learning Project""")
st.markdown("""
### AI-based Wi-Fi Human Presence Detection

This system detects human movement using Wi-Fi RSS signals and Machine Learning.
""")

st.title("Wi-Fi Human Presence Detection")
st_autorefresh(interval=3000, key="refresh")

st.info("🔄 Live Monitoring Enabled (Refresh every 3 seconds)")

st.write("Upload an RSS CSV file")

uploaded_file = st.file_uploader("Choose RSS CSV", type=["csv"])

if "history" not in st.session_state:
    st.session_state.history = []

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.write("Preview of uploaded file")
    st.dataframe(df.head())
    st.subheader("Dataset Information")

    st.write("Rows:", df.shape[0])
    st.write("Columns:", df.shape[1])
    st.subheader("RSS Signal Graph")
    

    fig, ax = plt.subplots(figsize=(10, 4))

    for col in df.columns:
        ax.plot(df[col], label=col)

    ax.set_xlabel("Sample Number")
    ax.set_ylabel("RSS Value")
    ax.set_title("RSS Signal Variation")
    ax.legend()  

    st.pyplot(fig)
 
    # Calculate Mean
    mean_1 = df.iloc[:, 0].mean()
    mean_2 = df.iloc[:, 1].mean()
    mean_3 = df.iloc[:, 2].mean()
    mean_4 = df.iloc[:, 3].mean()

    # Calculate Standard Deviation
    std_1 = df.iloc[:, 0].std()
    std_2 = df.iloc[:, 1].std()
    std_3 = df.iloc[:, 2].std()
    std_4 = df.iloc[:, 3].std()

# Show extracted features
    st.subheader("Extracted Features")

    feature_df = pd.DataFrame({
        "Feature": [
            "Mean 1", "Mean 2", "Mean 3", "Mean 4",
            "Std 1", "Std 2", "Std 3", "Std 4"
        ],
        "Value": [
            mean_1, mean_2, mean_3, mean_4,
            std_1, std_2, std_3, std_4
        ]
    })

    st.dataframe(feature_df)
    report = feature_df.to_csv(index=False)

    st.download_button(
       "📥 Download Feature Report",
       report,
       file_name="feature_report.csv",
       mime="text/csv"
    )
    st.subheader("Feature Analysis")

    fig2, ax2 = plt.subplots(figsize=(10,4))

    ax2.bar(feature_df["Feature"], feature_df["Value"])

    ax2.set_xlabel("Features")
    ax2.set_ylabel("Value")
    ax2.set_title("Extracted Feature Values")

    plt.xticks(rotation=45)

#3D Visualisation
    st.pyplot(fig2)
    st.subheader("3D Wi-Fi Signal Visualization")

    plot_df = df.copy()
    plot_df["Sample"] = range(len(plot_df))

    fig3d = px.scatter_3d(
        plot_df,
        x="Sample",
        y=plot_df.columns[0],
        z=plot_df.columns[1],
        color=plot_df.columns[2],
        size=abs(plot_df[plot_df.columns[3]]),
        title="3D Wi-Fi RSS Signal Distribution"
       )
    st.plotly_chart(fig3d, use_container_width=True)
    st.subheader("🏠 Estimated Room Layout")

    fig_room = go.Figure()

# Room boundary
    fig_room.add_shape(
    type="rect",
    x0=0,
    y0=0,
    x1=10,
    y1=10,
    line=dict(color="black", width=4),
    fillcolor="rgba(230,230,230,0.1)"
    )
# 🚪 Door 

    fig_room.add_shape(

        type="line",

        x0=4,

        y0=0,

        x1=6,

        y1=0,

        line=dict(color="brown", width=8)

    )
# Table
    fig_room.add_shape(
        type="rect",
        x0=3,
        y0=4,
        x1=5,  
        y1=6,
        fillcolor="lightblue",
        line=dict(color="blue")
    )

# Sofa
    fig_room.add_shape(
        type="rect",
        x0=7,
        y0=2,
        x1=9,
        y1=3.5,   
        fillcolor="green",
        line=dict(color="darkgreen")
     )
#ADD WIFI ROUTER    
    fig_room.add_trace(
    go.Scatter(
        x=[5],
        y=[9],
        mode="markers+text",
        marker=dict(size=18, color="green"),
        text=["📶 Router"],
        textposition="bottom center",
        name="Router"
    )
    )
#ADD Window
    fig_room.add_shape(
    type="line",
    x0=10,
    y0=3,
    x1=10,
    y1=7,
    line=dict(color="skyblue", width=6)
    )
#ADD TV
    fig_room.add_shape(
    type="rect",
    x0=7,
    y0=4,
    x1=8.5,
    y1=5.5,
    fillcolor="black",
    line=dict(color="gray")
    )
#ADD Motion Radius
    x = min(max(abs(mean_1) * 8, 1), 9)
    y = min(max(abs(mean_2) * 8, 1), 9)
    fig_room.add_shape(
    type="circle",
    x0=x-0.6,
    y0=y-0.6,
    x1=x+0.6,
    y1=y+0.6,
    line=dict(color="red"),
    fillcolor="rgba(255,0,0,0.15)"
    )
    

# Wi-Fi anchors
    anchors = {
    "A1": (1, 9),
    "A2": (9, 9),
    "A3": (1, 1),
    "A4": (9, 1)
    }

    for name, (ax, ay) in anchors.items():
        fig_room.add_trace(go.Scatter(
            x=[ax],
            y=[ay],
            mode="markers+text",
            marker=dict(size=15, color="blue"),
            text=[name],
            textposition="top center",
         name=name
        ))

# Estimated position(PERSON LOCATION)
    x = min(max(abs(mean_1) * 8, 1), 9)
    y = min(max(abs(mean_2) * 8, 1), 9)

    fig_room.add_trace(
    go.Scatter(
        x=[x],
        y=[y],
        mode="markers+text",
        marker=dict(size=30, color="red"),
        text=["🚶 Human"],
        textposition="top center",
        name="Person"
    )
)

   

    fig_room.update_layout(
        title="Estimated Human Position",
        xaxis=dict(range=[0,10], title="Room Width"),
        yaxis=dict(range=[0,10], title="Room Length"),
        height=550
    )

    st.plotly_chart(fig_room, use_container_width=True)
    
    # ⬇️ Dashboard statics

    st.subheader("Dashboard Statistics")

    col1, col2, col3, col4 = st.columns(4)

    total = len(st.session_state.history)
    movement = sum(1 for h in st.session_state.history if h["Prediction"] == "Movement")
    no_movement = total - movement

    avg_conf = (
    sum(float(h["Confidence"].replace("%", "")) for h in st.session_state.history) / total
    if total > 0 else 0
    )

    col1.metric("Total Detections", total)
    col2.metric("Movement", movement)
    col3.metric("No Movement", no_movement)
    col4.metric("Avg Confidence", f"{avg_conf:.1f}%")

    # ⬇️ ADD THE HEATMAP HERE

    st.subheader("RSS Correlation Heatmap")

    corr = df.corr()

    fig_heat, ax = plt.subplots(figsize=(6,5))

    heatmap = ax.imshow(corr, cmap="coolwarm")

    ax.set_xticks(range(len(corr.columns)))
    ax.set_yticks(range(len(corr.columns)))

    ax.set_xticklabels(corr.columns, rotation=45)
    ax.set_yticklabels(corr.columns)

    plt.colorbar(heatmap)

    st.pyplot(fig_heat)

    # Prediction
    features = np.array([[mean_1, mean_2, mean_3, mean_4,
                          std_1, std_2, std_3, std_4]])

    prediction = model.predict(features)
    from datetime import datetime

    confidence = model.predict_proba(features)[0].max() * 100

    st.session_state.history.append({
    "Time": datetime.now().strftime("%H:%M:%S"),
    "Prediction": "Movement" if prediction[0] == 1 else "No Movement",
    "Confidence": f"{confidence:.2f}%"
    })
    # ⬇️ Current Detection Status

    st.subheader("Current Detection Status")

    status_col1, status_col2 = st.columns(2)

    status_col1.metric(

    "Current Status",

    "🟢 Movement" if prediction[0] == 1 else "🔴 No Movement"

    )

    status_col2.metric(

    "Confidence",

    f"{confidence:.2f}%"

    )

    if hasattr(model, "predict_proba"):
        

        st.subheader("Prediction Confidence")
        st.progress(int(confidence))
        st.write(f"Confidence: {confidence:.2f}%")

    if prediction[0] == 1:
        st.success("🟢 Movement Detected")
    else:
        st.error("🔴 No Movement Detected")

        st.subheader("Detection History")

if st.session_state.history:
    history_df = pd.DataFrame(st.session_state.history)
    st.dataframe(history_df, use_container_width=True)
# 👇 ADD THE FOOTER HERE
st.markdown("---")
st.caption("Developed by Khush Prajapati | Wi-Fi Human Presence Detection")    