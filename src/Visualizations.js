import React from "react";
import { Bar, Pie } from "react-chartjs-2";
import {
  Chart as ChartJS,
  ArcElement,
  Tooltip,
  Legend,
  CategoryScale,
  LinearScale,
  BarElement,
} from "chart.js";

// Register the Chart.js components
ChartJS.register(ArcElement, Tooltip, Legend, CategoryScale, LinearScale, BarElement);

const Visualizations = ({ sentiment, topics }) => {
  // Data for the sentiment pie chart.
  // Assuming sentiment data looks like: { label: "POSITIVE", score: 0.95 }
  // For a richer visualization, you might compare multiple sentiments if available.
  const sentimentData = {
    labels: ["Sentiment"],
    datasets: [
      {
        label: "Sentiment Score",
        data: [sentiment ? sentiment.score * 100 : 0, sentiment ? (1 - sentiment.score) * 100 : 0],
        backgroundColor: ["#36A2EB", "#FF6384"],
        hoverBackgroundColor: ["#36A2EB", "#FF6384"],
      },
    ],
  };

  // Data for topics bar chart.
  // If topics is an array like ["innovation", "growth", "market"],
  // we can count their frequencies if needed or simply display each topic.
  // Here we will simply display a bar for each topic with a dummy frequency.
  // For a real-world scenario, consider processing the reports to get topic frequency counts.
  const topicCounts = topics.reduce((acc, topic) => {
    acc[topic] = (acc[topic] || 0) + 1;
    return acc;
  }, {});

  const topicData = {
    labels: Object.keys(topicCounts),
    datasets: [
      {
        label: "Topic Frequency",
        data: Object.values(topicCounts),
        backgroundColor: "rgba(75,192,192,0.6)",
        borderColor: "rgba(75,192,192,1)",
        borderWidth: 1,
      },
    ],
  };

  return (
    <div className="visualizations">
      <h3>Sentiment Analysis</h3>
      <Pie data={sentimentData} />

      <h3>Extracted Topics</h3>
      <Bar data={topicData} options={{ responsive: true, maintainAspectRatio: false }} style={{ height: "300px" }} />
    </div>
  );
};

export default Visualizations;
