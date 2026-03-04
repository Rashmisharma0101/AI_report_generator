import pandas as pd
import numpy as np
import streamlit as st
from groq import Groq

from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, ListFlowable, ListItem
from reportlab.lib.styles import getSampleStyleSheet
from reportlab.lib.pagesizes import A4
from reportlab.lib.units import inch
import io

st.title("AI-Powered Business Report Generator")

file = st.file_uploader("Upload File", type = ["csv"])

client = Groq(api_key=st.secrets["GROQ_API_KEY"])
if file:
	df = pd.read_csv(file)
	st.subheader("Raw Data  Preview")
	
	st.write("Rows in dataframe", df.shape[0])
	st.write("Columns in dataframe", df.shape[1])

	st.write(df.describe())
	st.write(df.dtypes)
	st.write("Missing Values per column")
	st.write(df.isnull().sum())
	
	df=  df.drop_duplicates()
	
	numeric_columns = df.select_dtypes(include = "number").columns
	categorical_columns = df.select_dtypes(include = "object").columns
	date_cols = df.select_dtypes(include=["datetime64"]).columns

	df[numeric_columns] = df[numeric_columns].fillna(df[numeric_columns].median())

	st.success("Basic Cleaning completed")

	
	st.write("Detected KPI")

	for col in numeric_columns:
		col_lower = col.lower()

		if "revenue" in col_lower or "sales" in col_lower or "amount" in col_lower:
			st.metric(label = f"Total {col}", value = round(df[col].sum(), 2))
			st.metric(label = f"Average {col}", value = df[col].mean())
		else:
			total = df[col].sum()
			avg = df[col].mean()

		

	for col in categorical_columns:
		st.metric(label = f"Unique {col}", value = df[col].nunique())



	st.metric("Total Rows" , df.shape[0])
	st.metric("Total columns", df.shape[1])


	st.subheader("Automated Exploratory Data Analysis")

	import matplotlib.pyplot as plt

	for col in numeric_columns:
		col_lower = col.lower()
		if col not in date_cols and "id" not in col.lower():
			st.subheader(f"Distribution of {col}")

			fig, ax = plt.subplots()
			ax.hist(df[col].dropna(), bins = 10)
			ax.set_title(f"{col} Distribution")
		
			st.pyplot(fig)


	for col in categorical_columns:
		if "date" not in col.lower():
			st.subheader(f"Top Categories in {col}")
    
			top_values = df[col].value_counts().head(10)
    
			fig, ax = plt.subplots()
			ax.bar(top_values.index.astype(str), top_values.values)
			ax.set_xticklabels(top_values.index.astype(str), rotation=45)
    
			st.pyplot(fig)

	

	if len(date_cols) > 0:
		date_col = date_cols[0]
		for col in numeric_columns:
			st.subheader(f"Trend of {col} over Time")
        
			df_sorted = df.sort_values(by=date_col)
        
			fig, ax = plt.subplots()
			ax.plot(df_sorted[date_col], df_sorted[col])
        
			st.pyplot(fig)


	def prepare_context(df):
		sample_data = df.head(5).to_string()
		columns = ", ".join(df.columns)
		shape = df.shape
    
		context = f"""
		Dataset Overview:
		Rows: {shape[0]}
		Columns: {shape[1]}
		Column Names: {columns}
    
		Sample Data:
		{sample_data}
    
		Generate a professional executive business summary.
		Highlight key insights, patterns, and possible business implications.


		Generate bullet points as Key Insights from the analysed data from this dataframe after creating charts
		"""
    			
		return context

	def generate_ai_summary(df):
		context = prepare_context(df)
    
		response = client.chat.completions.create(
		model="llama-3.1-8b-instant",
		messages=[
            	{"role": "user", "content": context}],
        	temperature=0.3)

		return response.choices[0].message.content


	st.subheader("AI executive Summary")
	if st.button("Generate AI insights"):
		with st.spinner("Analyzing Dataset"):
			st.session_state.ai_summary = generate_ai_summary(df)
	if "ai_summary" in st.session_state:
		st.write(st.session_state.ai_summary)

def generate_pdf_report(df, ai_summary):
		
	buffer = io.BytesIO()
	doc = SimpleDocTemplate(buffer, pagesize=A4)
	elements = []

	styles = getSampleStyleSheet()
	elements.append(Paragraph("AI-Powered Business Report", styles["Heading1"]))
	elements.append(Spacer(1, 0.3 * inch))

	# Dataset Overview
	overview_text = f"""
	<b>Dataset Overview</b><br/>
	Total Rows: {df.shape[0]}<br/>
	Total Columns: {df.shape[1]}<br/>
	"""
	elements.append(Paragraph(overview_text, styles["Normal"]))
	elements.append(Spacer(1, 0.3 * inch))

	# KPI Section
	elements.append(Paragraph("<b>Key Metrics</b>", styles["Heading2"]))
	elements.append(Spacer(1, 0.2 * inch))

	numeric_cols = df.select_dtypes(include="number").columns

	kpi_list = []
	for col in numeric_cols:
		total = df[col].sum()
		avg = df[col].mean()
		kpi_text = f"{col}: Total = {total:,.2f}, Average = {avg:,.2f}"
		kpi_list.append(ListItem(Paragraph(kpi_text, styles["Normal"])))

	elements.append(ListFlowable(kpi_list))
	elements.append(Spacer(1, 0.4 * inch))

	# AI Summary Section
	elements.append(Paragraph("<b>AI Executive Summary</b>", styles["Heading2"]))
	elements.append(Spacer(1, 0.2 * inch))

	formatted_summary = ai_summary.replace("\n", "<br/>")
	elements.append(Paragraph(formatted_summary, styles["Normal"]))

	doc.build(elements)

	buffer.seek(0)
	return buffer

st.subheader("📄 Download Business Report")

if "ai_summary" in st.session_state:
	if st.button("Generate PDF Report"):

		pdf_file = generate_pdf_report(df, st.session_state.ai_summary)
		st.download_button(
		label="Download AI Business Report",
		data=pdf_file,
		file_name="AI_Business_Report.pdf",
		mime="application/pdf"
    )





