# 6) Calculate mean and standard deviation of Transaction Amount
mean_amount = data['Transaction_Amount'].mean()
std_amount = data['Transaction_Amount'].std()


# 7)Define the anomaly threshold (2 standard deviations from the mean)
anomaly_threshold = mean_amount + 2 * std_amount


# 8)Flag anomalies
data['Is_Anomaly'] = data['Transaction_Amount'] > anomaly_threshold


# 9)Scatter plot of Transaction Amount with anomalies highlighted
fig_anomalies = px.scatter(data, x='Transaction_Amount', y='Average_Transaction_Amount',
color='Is_Anomaly', title='Anomalies in Transaction Amount')
fig_anomalies.update_traces(marker=dict(size=12),
selector=dict(mode='markers', marker_size=1))
fig_anomalies.show()


# 10)Calculate the number of anomalies
num_anomalies = data['Is_Anomaly'].sum()


# 11)Calculate the total number of instances in the dataset
total_instances = data.shape[0]


# 12)Calculate the ratio of anomalies
anomaly_ratio = num_anomalies / total_instances
print(anomaly_ratio)


