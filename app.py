# -*- coding: utf-8 -*-
"""
医疗问诊智能体
全部功能：药品查询 / 体检解读 / 症状问诊 / 病历整理
全部使用 LangChain + 通义千问 大模型
页面布局完全保留原版不变
"""
import streamlit as st
from datetime import datetime as dt
from dotenv import load_dotenv

# ========== 固定按你指定的导入格式 ==========
from langchain_community.chat_models.tongyi import ChatTongyi
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# 加载环境变量
load_dotenv()

# 初始化通义千问
llm = ChatTongyi(model="qwen-turbo", temperature=0.3)
parser = StrOutputParser()

# ---------------------- 各功能专属提示词模板 ----------------------
# 1.药品查询
drug_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业药师，给出药品通用名、适应症、用法用量、禁忌、不良反应、注意事项，条理清晰，不要虚假信息。"),
    ("user", "帮我查询药品：{query}")
])
drug_chain = drug_prompt | llm | parser

# 2.体检报告解读
report_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业体检解读医生，逐条分析指标高低、正常范围、临床意义、生活建议，语言通俗易懂，不做确诊，只做健康参考。"),
    ("user", "帮我解读这份体检报告：{query}")
])
report_chain = report_prompt | llm | parser

# 3.症状问诊
symptom_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是专业健康问诊助手，分析症状原因、给出居家处理建议、饮食作息注意事项，明确何时需要就医，不做疾病确诊。"),
    ("user", "我的症状：{query}")
])
symptom_chain = symptom_prompt | llm | parser

# 4.病历整理
record_prompt = ChatPromptTemplate.from_messages([
    ("system", "你是门诊病历整理助手，根据姓名、年龄、症状、既往史，整理成规范简洁的门诊病历格式，条理工整。"),
    ("user", "姓名：{name}，年龄：{age}，主要症状：{symptom}，既往史过敏史：{history}，帮我整理标准病历")
])
record_chain = record_prompt | llm | parser

# ---------------------- 工具函数封装 ----------------------
def get_drug_info(query):
    return drug_chain.invoke({"query": query})

def get_report_analysis(query):
    return report_chain.invoke({"query": query})

def get_symptom_answer(query):
    return symptom_chain.invoke({"query": query})

def get_medical_record(name, age, symptom, history):
    return record_chain.invoke({
        "name": name,
        "age": age,
        "symptom": symptom,
        "history": history
    })

# ===================== Streamlit 主界面（完全原版不变） =====================
def main():
    st.set_page_config(page_title="医疗问诊智能体", page_icon="🏥", layout="wide")
    st.title("🏥 医疗问诊智能体")
    st.subheader("智能问诊 | 药品查询 | 体检报告解读 | 健康咨询")

    menu = ["🏠 首页", "💊 药品查询", "📋 体检报告解读", "💬 症状问诊", "📝 病历整理"]
    choice = st.sidebar.selectbox("功能菜单", menu)

    if choice == "🏠 首页":
        st.success("欢迎使用医疗问诊智能体！请使用左侧菜单选择功能")
        st.markdown("""
        ### 支持功能
        ✅ 大模型智能药品查询
        ✅ 大模型体检报告一键解读
        ✅ AI 智能症状问诊咨询
        ✅ AI 自动规范整理门诊病历
        """)

    elif choice == "💊 药品查询":
        st.subheader("💊 药品信息查询")
        drug_name = st.text_input("请输入药品名称", placeholder="例如：布洛芬、阿莫西林")
        if st.button("查询药品信息") and drug_name:
            with st.spinner("AI药师查询中..."):
                res = get_drug_info(drug_name)
                st.markdown(res)

    elif choice == "📋 体检报告解读":
        st.subheader("📋 体检报告智能解读")
        report_text = st.text_area("请粘贴你的体检报告文本", height=200, placeholder="粘贴包含指标和数值的报告内容")
        if st.button("开始解读") and report_text:
            with st.spinner("AI医生解析报告中..."):
                res = get_report_analysis(report_text)
                st.markdown(res)

    elif choice == "💬 症状问诊":
        st.subheader("💬 AI 智能症状问诊")
        question = st.text_input("描述你的症状或问题", placeholder="例如：发烧了怎么办、咳嗽不停、胃痛")
        if st.button("获取AI解答") and question:
            with st.spinner("AI思考中..."):
                ans = get_symptom_answer(question)
                st.info(ans)

    elif choice == "📝 病历整理":
        st.subheader("📝 AI 智能病历整理")
        name = st.text_input("姓名")
        age = st.number_input("年龄", min_value=0, max_value=120, step=1)
        symptom = st.text_area("主要症状描述")
        history = st.text_area("既往病史/过敏史")
        if st.button("AI生成病历"):
            with st.spinner("AI整理病历中..."):
                record = get_medical_record(name, age, symptom, history)
                st.code(record, language="text")
                now = dt.now().strftime("%Y-%m-%d %H:%M")
                st.download_button("下载病历", record, file_name=f"病历_{now}.txt")

if __name__ == "__main__":
    main()