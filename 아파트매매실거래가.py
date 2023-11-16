import requests
from bs4 import BeautifulSoup
import sys
import io
import pandas as pd
from pandas import Series, DataFrame
import re
sys.stdout = io.TextIOWrapper(sys.stdout.detach(), encoding = 'utf-8')
sys.stderr = io.TextIOWrapper(sys.stderr.detach(), encoding = 'utf-8')

def City():
    Code=[]
    Four=[]
    with open('C:/Users/User/Desktop/파이썬/법정동코드 전체자료/법정동코드 전체자료.txt') as data:
        lines=data.readlines()
    for row in lines :
        row2=row.split()
        Code.append(row2[0])
    s1=set(Code)
    l1=list(s1)
    for i in l1 :
        Four.append(i[0:5])
    s2=set(Four)
    l2=list(s2)
    l2.sort()
    return l2

def Service(serviceKey,LAWD_CD,DEAL_YMD) :
    url='http://openapi.molit.go.kr/OpenAPI_ToolInstallPackage/service/rest/RTMSOBJSvc/getRTMSDataSvcAptTradeDev?serviceKey='\
     + serviceKey + '&pageNo=1&numOfRows=999'  +\
    '&LAWD_CD=' + str(LAWD_CD) +'&DEAL_YMD=' + str(DEAL_YMD)
    dic={}
    Trade_Amount=[]
    Construction_Year=[]
    Year=[]
    Road_Name=[]
    Road_Name_Main_Code=[]
    Road_Name_Sub_Code=[]
    Road_Name_City_Gungu_Code=[]
    Road_Name_Serial_Num=[]
    Road_Name_Up_Down_Code=[]
    Road_Name_Code=[]
    City_Name=[]
    City_Name_Main_Code=[]
    City_Name_Sub_Code=[]
    City_Name_City_Gungu_Code=[]
    City_Name_Um_Dong_Code=[]
    City_Name_parcel_Code=[]
    Apart_Name=[]
    Month=[]
    Day=[]
    Serial_Num=[]
    Area_Size=[]
    parcel_Code=[]
    Zone_Code=[]
    Floor=[]
    result=requests.get(url)
    html=result.text
    soup = BeautifulSoup(html, "html.parser")
    item=soup.find_all('item')
    for i in item  :
        b=i.text
        c=re.split('<.*?>', b)
        del c[0]

        if len(c)==22:
            c.insert(8,'0')
            c.insert(19,'')
        elif len(c)==23 :
            if c[8]=='0':
                c.insert(19,'')
            else:
                c.insert(8,'0')
        elif len(c)==18 :
            c.insert(3,'')
            c.insert(4,'')
            c.insert(5,'')
            c.insert(6,'')
            c.insert(9,'')
            c.insert(19,'')
        elif len(c)==24:
            pass
        else:
            continue
        Trade_Amount.append(c[0])
        Construction_Year.append(c[1])
        Year.append(c[2])
        Road_Name.append(c[3])
        Road_Name_Main_Code.append(c[4])
        Road_Name_Sub_Code.append(c[5])
        Road_Name_City_Gungu_Code.append(c[6])
        Road_Name_Serial_Num.append(c[7])
        Road_Name_Up_Down_Code.append(c[8])
        Road_Name_Code.append(c[9])
        City_Name.append(c[10])
        City_Name_Main_Code.append(c[11])
        City_Name_Sub_Code.append(c[12])
        City_Name_City_Gungu_Code.append(c[13])
        City_Name_Um_Dong_Code.append(c[14])
        City_Name_parcel_Code.append(c[15])
        Apart_Name.append(c[16])
        Month.append(c[17])
        Day.append(c[18])
        Serial_Num.append(c[19])
        Area_Size.append(c[20])

    dic['거래금액']=Trade_Amount
    dic['건축년도']=Construction_Year
    dic['년']=Year
    dic['도로명']=Road_Name
    dic['도로명건물본번호코드']=Road_Name_Main_Code
    dic['도로명건물부번호코드']=Road_Name_Sub_Code
    dic['도로명시군구번호코드']=Road_Name_City_Gungu_Code
    dic['도로명일련번호코드']=Road_Name_Serial_Num
    dic['도로명지상지하코드']=Road_Name_Up_Down_Code
    dic['도로명코드']=Road_Name_Code
    dic['법정동']=City_Name
    dic['법정동본번코드']=City_Name_Main_Code
    dic['법정동부번코드']=City_Name_Sub_Code
    dic['법정동시군구코드']=City_Name_City_Gungu_Code
    dic['법정동읍면동코드']=City_Name_Um_Dong_Code
    dic['법정동지번코드']=City_Name_parcel_Code
    dic['아파트']=Apart_Name
    dic['월']=Month
    dic['일']=Day
    dic['일련번호']=Serial_Num
    dic['전용면적']=Area_Size

    dataframe=pd.DataFrame(dic)
    return dataframe

#serviceKey='JC6P4YKfsr3eVCBRBSuoINLSJjslpwls6JwDy%2FgQv112ce%2FWOj5LxFHdEMnkc%2Fk2OOdMbwmWjI0XX7UsFsPL1w%3D%3D'
#serviceKey='9V%2BBGD8pGBpGmARKZnvFVLkBUvLsLjZ4AleBaew3JCJrcrLVHGoaco4ZEVSa46YD8ZJQYx7Ivw2Kmg%2BJ796Zqg%3D%3D'
#serviceKey='%2B%2BvKgj%2B9hR8rtAlXhoEABZVkN9OXxvTYaUzYpj4jS2XK8o0ILePEaKA4xv3XzJ%2BLPySoG2PegrMmhPvhnRBKEQ%3D%3D'
#serviceKey='LAmrcqhmnZuHwXZL%2B1xqqbmIoyJf79YWM49AXw7%2F245RL7t%2FzTMtn5%2BiLk2woXonHpoxN0Uu6EnzNW44q3%2BENA%3D%3D'
serviceKey='HRfCg3kvPvk2WjxfGKCfe%2FcdcyEerVFYjeOVVM16MepitLQmy5ovTvblgTm%2BXkpksxPPid496OuNVch691ZXmA%3D%3D'
#serviceKey='QrZ6JP7souHUBiIFSz6vvNG2LZcf3wjLrwkHyCisVpLcdvAMO4a9T%2B3EBUOjzOgy3HZzIF4JUZ%2FBRJMYamE4UQ%3D%3D'
LAWD_CD=City()
DEAL_YMD=201904
# 12 8
def main() :
    dic={}
    dic_1=pd.DataFrame(dic)
    for j in LAWD_CD :
        a=Service(serviceKey,j,DEAL_YMD)
        result=pd.concat([dic_1,a],ignore_index=True)
        dic_1=result
    result.to_csv('C:/Users/User/Desktop/파이썬/data/2019/%d.csv' %DEAL_YMD,encoding='ms949',header=False, index=False)
main()
