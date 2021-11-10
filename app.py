
from pandas.core.tools.datetimes import to_datetime
import streamlit as st
import pandas as pd
import requests
import json
from st_radial import st_radial
import plotly.graph_objects as go
import plotly.express as px
from datetime import date, timedelta

#from datetime import datetime

from htbuilder import HtmlElement, div, ul, li, br, hr, a, p, img, styles, classes, fonts
from htbuilder.units import percent, px
from htbuilder.funcs import rgba, rgb
from streamlit.elements.utils import clean_text


###############################################################################################################################
###############################################################################################################################

###### set config of the page

st.set_page_config(
    page_title="Mining Stake",
    page_icon="💰",
    layout="wide",
    initial_sidebar_state="expanded")

#### hide menu and streamlit footer

hide_streamlit_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            </style>
            """
st.markdown(hide_streamlit_style, unsafe_allow_html=True) 

##### get data function

def get_data(link, colname1, colname2):

    response = requests.get(link)
    data = json.loads(response.text)
    data = data["values"]
    data_name = pd.DataFrame.from_dict(data)
    data_name['x'] = pd.to_datetime(data_name['x'], unit='s')
    data_name = data_name.sort_values(by = "x", ascending = False ) 
    data_name.columns = [colname1, colname2]

    return data_name

#start = date.today() - timedelta(days=14)
#end = date.today() 
###### load data function

def load_data(start, end):

    period = round((end - start).days/7)

    ### total fees in USD
    fees = "https://api.blockchain.info/charts/transaction-fees-usd?timespan="+str(period)+"weeks&rollingAverage=8hours&format=json"  
    fees_data = get_data(fees, "date", "fee_USD")

    ### fees in BTC
    fees_btc = "https://api.blockchain.info/charts/transaction-fees?timespan="+str(period)+"weeks&rollingAverage=8hours&format=json"
    fees_btc_data = get_data(fees_btc, "date", "fee_BTC")

    #### difficulty 
    difficulty = "https://api.blockchain.info/charts/difficulty?timespan="+str(period)+"weeks&rollingAverage=8hours&format=json"
    difficulty_data = get_data(difficulty, "date", "difficulty")

    #### exchange rate
    price = "https://api.blockchain.info/charts/market-price?timespan="+str(period)+"weeks&rollingAverage=8hours&format=json"
    price_data = get_data(price, "date", "price")

    ### joining data
    result = pd.merge(price_data, difficulty_data, on = "date")
    result = pd.merge(result, fees_btc_data, on = "date")
    result = pd.merge(result, fees_data, on = "date")

    ####  mutating columns
    result["date"] = result["date"].dt.date
    result["date"] = result["date"].apply(lambda x: x.strftime('%d/%m/%Y'))
    result["price"] = result["price"].apply(lambda x: int(x))
    result["difficulty"] = result["difficulty"].apply(lambda x: int(x))
    result["fee_USD"] = result["fee_USD"].apply(lambda x: int(x))
    result["fee_BTC"] = result["fee_BTC"].apply(lambda x: round(x,3))

    return result


#######  authentication function

def is_authenticated(password):
    return password == "1"

def is_authenticated_user(user):
    return user == "a"

def generate_login_block():
    block1 = st.sidebar.empty()
    block2 = st.sidebar.empty()
    block3 = st.sidebar.empty()

    return block1, block2, block3

def clean_blocks(blocks):
    for block in blocks:
        block.empty()

def login(blocks):
    blocks[0].markdown("""
            <style>
                input {
                    -webkit-text-security: disc;
                }
            </style>
        """, unsafe_allow_html=True)

    return blocks[1].text_input('Username'), blocks[2].text_input('Password')

page = ""

##### creating the sidebar

st.sidebar.title("Welcome to Mining Stake")

def main():
    page = st.sidebar.selectbox("Select a page", ["Home", "Statistics", "Mining Calculator"])

    if page == "":
        st.subheader("Please Log in")
        st.write("Enter your user name and passwor in the lateral bar in order to enter to the system")
        "------"

###############################################################################################################################
###############################################################################################################################

    if page == "Home":

        st.title("Let's know more about our business")

        st.header("")

        st.image("image_1.jpg", use_column_width=True)

        st.header("What do we do?")

        st.write("Our goal is to the dilever a quality service to our customers, we invite you to know more about our business model")

        #st.markdown("[![Follow](<https://img.shields.io/twitter/follow/><laloaldea>?style=social)](<https://www.twitter.com/><laloaldea>)", unsafe_allow_html=True)

        col0, col1, col2, col3, col4 = st.columns([1,4,4,4,1])
        ""

        with col0:
            ""

        with col1: 
            st.write("  Our suppliers")
            st.image("bitcoin.jpeg")
            ""
            st.write("Tenemos un up-time del 99%")
            st_radial("up-time", 99,  start_angle=0, end_angle=355)

        with col2: 
            st.write("How we add value")
            st.image("bitcoin.jpeg")
        
        with col3: 
            st.write("Your advantages")
            st.image("bitcoin.jpeg")

        with col4:
            ""


###############################################################################################################################
###############################################################################################################################


    if page == "Statistics":

        st.title("Welcome to our Statistics dashboard")

        st.write("In this page you can see both, last day and historical statistics about BitCoin")

        ""

        st.subheader("See hitorical statistics")

        ""

        with st.expander("Show hitorical statistics"):

            st.subheader("Select a range of dates")

            ""

            "Please, select a range of dates in order to get updated bitcoin information."

            ""

            min_date = date.today() - timedelta(days=14)
            today = date.today() 

            col1, col2, col3, col4 = st.columns([1,3,3,1])

            with col1:
                ""
            
            with col2:
            
                start = st.date_input("Start date", value=min_date, max_value=min_date)

            with col3:
            
                end = st.date_input("End date", value=today, min_value=today, max_value=today)
            
            with col4:
                ""
                ""
                #st.button("🔄")
            "--------"

            ###### Variables for indicators

            result = load_data(start, end)

            actual_price = int(result.iloc[1,1:2])
            last_price = int(result.iloc[-1,1:2])

            actual_difficulty = int(result.iloc[0,2:3])
            last_difficulty = int(result.iloc[-1,2:3])

            last_fee_bt_0 = float(result.iloc[0,3:4])
            last_fee_bt_1 = float(result.iloc[-1,3:4])

            last_fee_0 = int(result.iloc[0,4:5])
            last_fee_1 = int(result.iloc[-1,4:5])


            st.subheader("Key Indicator for the selected period data")

            ###### historical metrics 

            st.write("Calculating the key metrics for the period starting on ", start , "till ", end)

            ""

            col0, col1, col2, col3 = st.columns([2.5,3,3,1])

            with col0:
                ""
            with col1:
                st.metric("USD price variation: ", format(actual_price, ",d"), delta= format(actual_price - last_price, ",d"))
                ""
                st.metric("Fee variation in USD: ", format(last_fee_0, ",d"), delta= format(last_fee_0 - last_fee_1, ",d"))

            with col2:
                st.metric("Network difficulty: ", format(actual_difficulty, ".2e"), delta= format(actual_difficulty - last_difficulty, ".2e"))
                ""
                st.metric("Fee variation in BTC: ", last_fee_bt_0 , delta= last_fee_bt_0 - last_fee_bt_1)

            with col3:
                ""

            #st.dataframe(result, height= 1000, width=1000)

            "------"

            #### last day metrics plots show
            
            plot = result.sort_index(ascending=False)
            plot = plot.reset_index()

            st.subheader("Historical plots")

            st.write("Please, select a variable to visualize the historical information")

            selected = st.selectbox("Select the variable",["price", "difficulty", "fee_BTC", "fee_USD"])

            ""

            fig = go.Figure([go.Line(x=plot['date'], y=plot[selected])])

            fig.update_xaxes(
                rangeslider_visible = True)

            
            fig.update_layout(height=500, margin=dict(l=10, r=10, b=10, t=10))

            st.plotly_chart(fig,use_container_width=True, config= {'displayModeBar': False})    

            ###### historical data table

            fig = go.Figure(data=[go.Table(

                columnorder = [1,2,3,4,5],

                columnwidth = [800,800,800,800,800],

                header=dict(values=list(result.columns),
                    fill_color='paleturquoise',
                    align='center',
                    font_size = 20,
                    height = 40),

                cells=dict(values=[result.date, 
                    result.price.apply(lambda x: format(int(x),",d")), 
                    result.difficulty.apply(lambda x: format(int(x),",d")), 
                    result.fee_BTC, 
                    result.fee_USD.apply(lambda x: format(int(x),",d"))],
                    fill_color='lavender',
                    align='center',
                    font_size = 15,
                    height = 30)
                    )
            ])
            #### historical data table show 
            
            "------"

            st.subheader("Historical data")

            st.write("Showing the data for the period selected above")

            ""

            fig.update_layout(height=400, margin=dict(l=10, r=10, b=10, t=10))

            st.plotly_chart(fig, use_container_width=True, config= {'displayModeBar': False})



        ### last data available table

        "----"

        st.subheader("See last day statistics")

        ""

        with st.expander("Show last day statistics"):

            ###### current block reward value

            st.subheader("Current Block Reward")

            reward = "https://blockchain.info/q/bcperblock"

            response = requests.get(reward)

            last_reward = json.loads(response.text)

            st.write("The current block reaward is ", str(last_reward), "BitCoins per mined Block")

            "----"
                
            #### last day metrics

            actual_price = int(result.iloc[1,1:2])
            last_price = int(result.iloc[2,1:2])

            actual_difficulty = int(result.iloc[0,2:3])
            last_difficulty = int(result.iloc[1,2:3])

            last_fee_bt_0 = float(result.iloc[0,3:4])
            last_fee_bt_1 = float(result.iloc[1,3:4])

            last_fee_0 = int(result.iloc[2,4:5])
            last_fee_1 = int(result.iloc[1,4:5])

            st.subheader("Key Indicators for the last day")

            ###### historical metrics 

            st.write("Calculating the variation of the key metrics during the last day")
            ""

            col0, col1, col2, col3 = st.columns([2.5,3,3,1])

            with col0:
                ""
            with col1:
                st.metric("USD price variation: ", format(actual_price, ",d"), delta= format(actual_price - last_price, ",d"))
                ""
                st.metric("Fee variation in USD: ", format(last_fee_0, ",d"), delta= format(last_fee_0 - last_fee_1, ",d"))

            with col2:
                st.metric("Network difficulty: ", format(actual_difficulty, ".2e"), delta= format(actual_difficulty - last_difficulty, ".2e"))
                ""
                st.metric("Fee variation in BTC: ", round(last_fee_bt_0, 2), delta= round(last_fee_bt_0 - last_fee_bt_1, 2))

            with col3:
                ""
            "------"
            #### last day metrics table show

            fig = go.Figure(data=[go.Table(

            columnorder = [1,2,3,4,5],

            columnwidth = [800,800,800,800,800],

            header=dict(values=list(result.columns),
                fill_color='paleturquoise',
                align='center',
                font_size = 20,
                height = 40),

            cells=dict(values=[result.date.iloc[0], 
                format(result.price.iloc[0], ",d"), 
                format(result.difficulty.iloc[0], ",d"), 
                result.fee_BTC.iloc[0], 
                format(result.fee_USD.iloc[0], ",d")],
                fill_color='lavender',
                align='center',
                font_size = 15,
                height = 30)
                )
            ])

            st.subheader("Last day information")

            st.write("The information for the las day in presented below")
            ""

            fig.update_layout(height=100, margin=dict(l=10, r=10, b=10, t=10))

            st.plotly_chart(fig, use_container_width=True, config= {'displayModeBar': False})


        #fig = go.Bar(x= result.date ,y=result.price, showlegend = True)
        ""

        
        
####################################################################################################################################
###############################################################################################################################

    if page == "Mining Calculator":

        st.subheader("Welcome to the mining calculator")
        st.write("Select the values that you would like to purchase in order to proyect your costs and earnings")
        ""

        start = date.today() - timedelta(days=14)
        end = date.today() 

        result = load_data(start, end) #result table

        ### Admins Variables Zone

        trlln = 1000000000000
        secperday = 86400
        constant = 2**32

        last_difficulty = int(result.iloc[0,2:3]) #result table
        last_fee = float(result.iloc[0,3:4]) #result table

        ####### getting last reward

        reward = "https://blockchain.info/q/bcperblock"

        response = requests.get(reward)

        last_reward = json.loads(response.text)

        ##### calculator

        with st.form(key = "inputs" ):

            col1, col2 = st.columns(2)

            with col1:
                device = st.number_input("Quantity of devices working", step = 1, value = 1)
                hr = st.number_input("Average hash rate of the devices (Th/s)", step = 1, value = 81)

            with col2:
                power = st.number_input("Device power in Watts", step = 1, value = 3450)
                elect = st.number_input("Electricity cost in USD", step = 0.001, value = 0.033)

            uptime = st.slider('Uptime of the machines',min_value=0, max_value=100, value = 99)  

            if st.form_submit_button("Calculate the values"):

                ""

                col1, col2 = st.columns(2)

                with col1:

                    mined = (device*hr*trlln*secperday*(last_reward + last_fee/(24*6))/(last_difficulty*constant*99/100))

                    st.success("Projected BTC mined")

                    st.subheader(round(mined, 5))
                
                with col2: 
                    mined_usd = mined*int(result.iloc[0,1:2])

                    st.success("Equivalent to USD")

                    st.subheader(round(mined_usd,2))
                    ""

                col1, col2 = st.columns(2)

                with col1: 
                    cost_bc = ((device*(power/1000)*24*elect) + (mined_usd - (device*(power/1000)*24*elect))*0.2)

                    st.success("Mining Hotel Monthly Cost")
                    
                    st.subheader(round(cost_bc,2))
                    ""

                with col2:
                    service_fee = device*(power/1000)*24*0.097

                    st.success("Service Administration Fee")

                    st.subheader(round(service_fee,2))

                earnings = (mined_usd - cost_bc - service_fee)

                st.success("Your average monthly earings will be:")
                st.subheader(round(earnings,2))



login_blocks = generate_login_block()

user, password  = login(login_blocks)

if is_authenticated(password) and is_authenticated_user(user):
    
    clean_blocks(login_blocks)
    main()

elif password:
    st.info("Please enter a valid user or password")








