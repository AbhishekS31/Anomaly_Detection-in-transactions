# 1)Distribution of Transaction Amount
fig_amount = px.histogram(data, x='Transaction_Amount',
nbins=20,
title='Distribution of Transaction Amount')
fig_amount.show()


# 2)Transaction Amount by Account Type
fig_box_amount = px.box(data,
x='Account_Type',
y='Transaction_Amount',
title='Transaction Amount by Account Type')
fig_box_amount.show()


# 3)Average Transaction Amount vs. Age
fig_scatter_avg_amount_age = px.scatter(data, x='Age',y='Average_Transaction_Amount',color='Account_Type',
title='Average Transaction Amount vs. Age',
trendline='ols')
fig_scatter_avg_amount_age.show()


# 4)Count of Transactions by Day of the Week
fig_day_of_week = px.bar(data, x='Day_of_Week',
title='Count of Transactions by Day of the Week')
fig_day_of_week.show()


# Correlation Heatmap
numeric_data = data.select_dtypes(include='number')
correlation_matrix = numeric_data.corr()
fig_corr_heatmap = px.imshow(correlation_matrix, title='Correlation Heatmap')
fig_corr_heatmap.show()