import numpy as np
import pandas as pd
import plotly.express as px
import plotly.figure_factory as ff
import matplotlib.pyplot as plt
import seaborn as sns

import pycountry as pct

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression


data = pd.read_csv('DATA.csv') #Để file dataset cùng 1 folder với file code

#------------BẮT ĐẦU QUÁ TRÌNH TIỀN XỬ LÝ DỮ LIỆU CHO PHÂN TÍCH MÔ TẢ------------

# Tóm lược dữ liệu (Đo mức độ tập trung & mức độ phân tán)
description = data.describe()
mode = data.select_dtypes(include=['float64','int64']).mode().iloc[0]
mode.name = 'mode'
median = data.select_dtypes(include=['float64','int64']).median()
median.name = 'median'
description = pd.concat([description, mode.to_frame().T])

description = pd.concat([description, median.to_frame().T])
print(description)

# Kiểm tra tỷ lệ lỗi thiếu data
data_na = (data.isnull().sum() / len(data)) * 100
missing_data = pd.DataFrame({'Ty le thieu data': data_na})
print(missing_data)

# Kiểm tra data bị trùng
duplicated_rows_data = data.duplicated().sum()
print(f"\nSO LUONG DATA BI TRUNG LAP: {duplicated_rows_data}")
data = data.drop_duplicates()

#điền khuyết cột company
data['Company Name'] = data['Company Name'].fillna(data['Company Name'].mode().iloc[0])

# Chuyển đổi cột Salary từ tiền ấn sang USD với tỉ giá 80
data['Salary'] = data['Salary'] / 80

#chuyển tiền sang int
data['Salary'] = data['Salary'].astype('int64')

# Lưu DataFrame đã được cập nhật thành file CSV mới
data.to_csv("DATA_updated.csv", index=False)

print(data['Salary'].dtypes)

#-----------------PHAN TICH MO TA----------------------
data=pd.read_csv('DATA_updated.csv')

data_complete = data.describe(include='all')
print(data_complete)

# 3.3.1. Biểu đồ hình tròn phân bổ lĩnh vực công việc ngành công nghệ phần mềm (PTMT: Đơn biến - dữ liệu phi số)
job_category = data['Job Category'].value_counts().sort_values(ascending=True)
fig1 = px.pie(values=job_category.values,
              names=job_category.index,
              color=job_category.index,
              title="BIỂU ĐỒ HÌNH TRÒN PHÂN BỔ LĨNH VỰC LÀM VIỆC NGÀNH CÔNG NGHỆ PHẦN MỀM")
fig1.update_traces(textinfo='label+percent+value',
                    textposition='outside')
fig1.show()

# Xuất dữ liệu phân bổ lĩnh vực làm việc ra màn hình
print("Dữ liệu phân bổ lĩnh vực làm việc:")
print(job_category)

# 3.3.2. Biểu đồ hình tròn tình trạng việc làm ngành công nghệ phần mềm (PTMT: Đơn biến - dữ liệu phi số)
employment_status = data['Employment Status'].value_counts().sort_values(ascending=True)
fig1 = px.pie(values=employment_status.values,
              names=employment_status.index,
              color=employment_status.index,
              title="BIỂU ĐỒ HÌNH TRÒN TÌNH TRẠNG VIỆC LÀM NGÀNH CÔNG NGHỆ PHẦN MỀM")
fig1.update_traces(textinfo='label+percent+value',
                    textposition='outside')
fig1.show()

# Xuất dữ liệu phân bổ lĩnh vực làm việc ra màn hình
print("Dữ liệu phân bổ lĩnh vực làm việc:")
print(employment_status)

# 3.3.3. Biểu đồ hình tròn vai trò công việc ngành công nghệ phần mềm (PTMT: Đơn biến - dữ liệu phi số)
job_roles = data['Job Roles'].value_counts().sort_values(ascending=True)
fig1 = px.pie(values=job_roles.values,
              names=job_roles.index,
              color=job_roles.index,
              title="BIỂU ĐỒ HÌNH TRÒN VAI TRÒ CÔNG VIỆC NGÀNH CÔNG NGHỆ PHẦN MỀM")
fig1.update_traces(textinfo='label+percent+value',
                    textposition='outside')
fig1.show()

# Xuất dữ liệu phân bổ lĩnh vực làm việc ra màn hình
print("Dữ liệu phân bổ lĩnh vực làm việc:")
print(job_roles)

# 3.3.5. Biểu đồ displot dữ liệu lương tính theo USD (PTMT: Đơn biến - dữ liệu số)
fig2 = ff.create_distplot(hist_data=[data['Salary']],
                          group_labels=['Salary'],
                          bin_size=500,
                          curve_type='kde')
fig2.update_layout(xaxis_title='Lương (USD)',
                   yaxis_title='Tần suất (Đã hiệu chỉnh)',
                   title='BIỂU ĐỒ DISPLOT CỦA LƯƠNG (USD)')
fig2.show()

# 3.3.6. Biểu đồ boxplot phân bổ lương (USD) theo Location (PTMT: Đa biến (2) - dữ liệu hỗn hợp)
fig3 = px.box(data_frame=data,
              x='Location',
              y='Salary',
              color='Location',
              title='BIỂU ĐỒ BOXPLOT PHÂN BỔ LƯƠNG (USD) TÌNH TRẠNG VIỆC LÀM')
fig3.update_layout(xaxis_title='Location',
                   yaxis_title='Lương (USD)')
fig3.show()

# Xuất dữ liệu phân bổ lương theo theo địa điểm công ty việc làm
print("Dữ liệu phân bổ lương theo theo địa điểm công ty:")
print(data[['Location', 'Salary']])

# 3.3.7. Biểu đồ boxplot phân bổ lương (USD) theo tình trạng việc làm (PTMT: Đa biến (2) - dữ liệu hỗn hợp)
fig3 = px.box(data_frame=data,
              x='Employment Status',
              y='Salary',
              color='Employment Status',
              title='BIỂU ĐỒ BOXPLOT PHÂN BỔ LƯƠNG (USD) TÌNH TRẠNG VIỆC LÀM')
fig3.update_layout(xaxis_title='Employment Status',
                   yaxis_title='Lương (USD)')
fig3.show()

# Xuất dữ liệu phân bổ lương ttheo tình trạng việc làm ra màn hình
print("Dữ liệu phân bổ  theo tình trạng việc làm:")
print(data[['Employment Status', 'Salary']])

# 3.3.8. Biểu đồ boxplot phân bổ lương (USD) theo vai trò công việc (PTMT: Đa biến (2) - dữ liệu hỗn hợp)
fig3 = px.box(data_frame=data,
              x='Job Roles',
              y='Salary',
              color='Job Roles',
              title='BIỂU ĐỒ BOXPLOT PHÂN BỔ LƯƠNG (USD) TÌNH TRẠNG VIỆC LÀM')
fig3.update_layout(xaxis_title='Job Roles',
                   yaxis_title='Lương (USD)')
fig3.show()

# Xuất dữ liệu phân bổ lương theo chế độ làm việc ra màn hình
print("Dữ liệu phân bổ  theo vai trò công việc  :")
print(data[['Job Roles', 'Salary']])

# 3.3.9. Biểu đồ boxplot phân bổ lương (USD) theo lĩnh vực công việc (PTMT: Đa biến (2) - dữ liệu hỗn hợp)
fig3 = px.box(data_frame=data,
              x='Job Category',
              y='Salary',
              color='Job Category',
              title='BIỂU ĐỒ BOXPLOT PHÂN BỔ LƯƠNG (USD) TÌNH TRẠNG VIỆC LÀM')
fig3.update_layout(xaxis_title='Job Category',
                   yaxis_title='Lương (USD)')
fig3.show()

# Xuất dữ liệu phân bổ lương theo lĩnh vực công việc ra màn hình
print("Dữ liệu phân bổ theo lĩnh vực công việc:")
print(data[['Job Category', 'Salary']])

#bieu do cot 3.3.10
# Đọc dữ liệu từ file CSV
file_path = 'DATA_updated.csv'  # Thay đổi đường dẫn đến file CSV của bạn
df = pd.read_csv(file_path)

# Tạo biểu đồ cột
plt.figure(figsize=(12, 6))
sns.barplot(x='Job Roles', y='Salary', hue='Job Category', data=df)

# Thêm chú giải (legend)
plt.legend(title='Job Category', title_fontsize='15', fontsize='12', bbox_to_anchor=(1.05, 1), loc='upper left')

# Đặt tên trục và tiêu đề biểu đồ
plt.title('Salary based on Job Roles and Job Category', fontsize='18')
plt.xlabel('Job Roles', fontsize='15')
plt.ylabel('Salary', fontsize='15')

# Thêm chú thích về trục y (Salary)
plt.annotate('Salary is in USD', xy=(0.5, -0.15), ha='center', va='center', fontsize='12', color='gray')

# Hiển thị biểu đồ
plt.show()

#bieu do heatmap 3.3.10
# Đọc dữ liệu từ file CSV
file_path = 'DATA_updated.csv'  # Thay đổi đường dẫn đến file CSV của bạn
df = pd.read_csv(file_path)

# Tạo biểu đồ heatmap
plt.figure(figsize=(12, 6))
heatmap = sns.heatmap(df.pivot_table(index='Job Roles', columns='Job Category', values='Salary', aggfunc='mean'), cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)

# Thêm colorbar
cbar = heatmap.collections[0].colorbar
cbar.set_label('Mean Salary', rotation=270, labelpad=15, fontsize='12')

# Đặt tiêu đề và nhãn trục
plt.title('Salary based on Job Roles and Job Category', fontsize='18')
plt.xlabel('Job Category', fontsize='15')
plt.ylabel('Job Roles', fontsize='15')

plt.show()

#bieu do cot 3.3.11
# Tạo biểu đồ bar chart
plt.figure(figsize=(15, 8))
sns.barplot(x='Location', y='Salary', hue='Job Category', data=df, ci=None, palette='Set2')
plt.title('Bar Chart of Salary based on Location and Job Category', fontsize=18)
plt.xlabel('Location', fontsize=15)
plt.ylabel('Salary', fontsize=15)
plt.legend(title='Job Category', title_fontsize='12', fontsize='10', bbox_to_anchor=(1.05, 1), loc='upper left')
plt.show()

#bieu do heatmap 3.3.11
# Tạo biểu đồ heatmap
plt.figure(figsize=(14, 8))
heatmap = sns.heatmap(df.pivot_table(index='Location', columns='Job Category', values='Salary', aggfunc='mean'),
                      cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)
plt.title('Salary based on Location and Job Category', fontsize='18')
plt.xlabel('Job Category', fontsize='15')
plt.ylabel('Location', fontsize='15')

# Hiển thị biểu đồ
plt.show()


#bieu do cot 3.3.12
# Tạo biểu đồ cột dạng tách biệt (grouped bar chart)
plt.figure(figsize=(12, 8))
sns.barplot(x='Job Category', y='Salary', hue='Employment Status', data=df, ci=None)  # ci=None để không hiển thị khoảng tin cậy
plt.title('Salary based on Employment Status and Job Category', fontsize=18)
plt.xlabel('Job Category', fontsize=15)
plt.ylabel('Salary', fontsize=15)
plt.legend(title='Employment Status', title_fontsize='13', fontsize='11')
plt.show()

#bieu do heatmap 3.3.12
# Tạo biểu đồ heatmap
plt.figure(figsize=(15, 8))
heatmap_data = df.pivot_table(index='Job Category', columns='Employment Status', values='Salary', aggfunc='mean')
sns.heatmap(heatmap_data, cmap='YlGnBu', annot=True, fmt=".2f", linewidths=.5)
plt.title('Heatmap of Salary based on Employment Status and Job Category', fontsize=18)
plt.xlabel('Employment Status', fontsize=15)
plt.ylabel('Job Category', fontsize=15)

plt.show()


