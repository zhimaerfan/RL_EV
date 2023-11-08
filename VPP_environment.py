import os
import numpy as np
import pandas as pd
from random import randrange
import contextlib
from gym import Env
from gym.spaces import Box, Dict, MultiDiscrete

from elvis.simulate import simulate
from elvis.utility.elvis_general import create_time_steps
from elvis.utility.elvis_general import num_time_steps
from dateutil.relativedelta import relativedelta

import plotly.graph_objects as go
from plotly.subplots import make_subplots
import plotly.express as px

"""注意:所有的功率为正常功率kw,消耗的能量为功率kw*15min即0.25h为正常消耗的能量kWh;VPP=虚拟电厂Virtual Power Plant, ELVIS=EV充电架构仿真Electric Vehicle Charging Infrastructure Simulator;一共三组,一个是没有EV,一个是有EV正常充放,一个是RL控制EV的VPP"""
def VPP_Scenario_config(yaml_config):
    """
    Function to extrapolate the VPP simulation data from the Elvis YAML config file
    """
    # 先调用，返回VPP所有参数，再输入给下面的class VPPEnv：
    # VPP_config_file = VPP_Scenario_config(yaml_str)  # VPP应该指的是整个优化场景
    # 从场景配置文件中获取参数并保存在并返回simulation_param
    # D:\OneDrive - stu.neu.edu.cn\code\ubuntu\Car\RL_VPP_Thesis-main\data\config_builder
    # 从Elvis YAML配置文件中推断VPP模拟数据的功能
    start_date = yaml_config["start_date"]
    end_date = yaml_config["end_date"]
    resolution = yaml_config["resolution"]

    num_households_load = yaml_config["num_households_load"]
    av_max_households_load = yaml_config["av_max_households_load"]
    av_max_energy_price = yaml_config["av_max_energy_price"]
    # 额定功率max=18kW
    solar_power = yaml_config["rw_infrastructure"][0]["solar_inverter"]["max_power"]
    # 额定功率max=13kW
    wind_power = yaml_config["rw_infrastructure"][1]["wind_inverter"]["max_power"]

    # self.EV_types[0]["battery"]["capacity"]
    EV_types = yaml_config["vehicle_types"]
    # - battery: {
    #     capacity: 100,  # kWh
    #     efficiency: 1,
    #     max_charge_power: 150,
    #     min_charge_power: 0}
    #   brand: 'Tesla'
    #   model: 'Model S'
    #   probability: 1
    #
    # transformer_preload: 0

    # 基础设施-变压器-充电站数量
    charging_stations_n = len(yaml_config["infrastructure"]["transformers"][0]["charging_stations"])
    # 设置的每周充电事件
    EVs_n = yaml_config["num_charging_events"] #per week
    # 设置的每年充电事件
    EVs_n_max = int(EVs_n*52.2) #(52.14 weeks in a year)
    # 50%,百分数乘过100了
    EVs_mean_soc = yaml_config["mean_soc"]
    # SoC标准差 ±10%  乘过100了
    EVs_std_deviation_soc = yaml_config["std_deviation_soc"]
    # 平均停车时间h  #hours
    mean_park = yaml_config["mean_park"]
    # 停车时间h标准差±1h
    std_deviation_park =yaml_config["std_deviation_park"]
    # 汽车负载充电功率上限，初始为0 ，来一个车就累加  # 额定功率max=45kW
    EV_load_max = 0
    # 电动汽车负载 额定功率 ，来一个车就累加3.7kW
    EV_load_rated = 0
    # ["基础设施/架构"]["变压器"][0]["充电站"]
    # 计算所有4个充电站最大功率和额定功率
    for charging_point in yaml_config["infrastructure"]["transformers"][0]["charging_stations"]:
        # 额定功率max=11kW，求4个充电站的最大充电功率11+11+11+11=44
        EV_load_max = EV_load_max + charging_point["max_power"]#kW
        # 3.7kW  # charging_point  3.7+3.7+3.7+3.7=14.8
        EV_load_rated = EV_load_rated + charging_point["rated_power"]#kW
    # 充电桩充电功率下限=1
    EV_load_min = yaml_config["infrastructure"]["transformers"][0]["min_power"]
    # ？  家庭最大功率max=10kW    # 10kw
    houseRWload_max = av_max_households_load

    # 从yaml_config读取后，SoC转换成百分比，保存在simulation_param中供以后使用
    simulation_param = {
        "start_date": start_date,
        "end_date": end_date,
        "resolution": resolution,

        "num_households": num_households_load,
        "solar_power": solar_power, #kw
        "wind_power": wind_power, #kw
        "EV_types": EV_types,

        # charging_stations里一共4个charging_point
        "charging_stations_n": charging_stations_n,
        "EVs_n": EVs_n,
        "EVs_n_max": EVs_n_max,
        "mean_park": mean_park, #hours
        "std_deviation_park": std_deviation_park, #std in hours
        # 抵达时的电池SoC%
        "EVs_mean_soc": EVs_mean_soc*100, #% battery on arrival
        # SoC标准差%
        "EVs_std_deviation_soc": EVs_std_deviation_soc*100, #Std in kWh
        "EV_load_max": EV_load_max, #kW
        "EV_load_rated": EV_load_rated, #kW
        "EV_load_min": EV_load_min, #kW
        "houseRWload_max": houseRWload_max, #kW
        #   0.13
        "av_max_energy_price": av_max_energy_price #€/kWh
    }
    return simulation_param


class VPPEnv(Env):
    """
    基于openAI gym Env class的VPP环境类. 
    VPP_data_input_path: 测试、训练、验证的数据输入路径
    VPP environment class based on the openAI gym Env class
    """
    # 后面两个参数有重复
    def __init__(self, VPP_data_input_path, elvis_config_file, simulation_param):
        """
        初始化函数，设置所有的模拟参数、变量、动作和状态空间。
        调用：
        在VPP_simulator.ipynb中：
        yaml_str = yaml.full_load(file)
        elvis_config_file = ScenarioConfig.from_yaml(yaml_str)  # elvis_config_file输入EV参数，调用Elvis返回的EV分布
        VPP_config_file = VPP_Scenario_config(yaml_str)
        env = VPPEnv(VPP_testing_data_input_path, elvis_config_file, VPP_config_file)
        env.plot_Dataset_autarky()

        # 从路径加载VPP数据Loading VPP data from path
        Initialization function to set all the simulation parameters, variables, action and state spaces.
        """
        #Loading VPP data from path
        VPP_data = pd.read_csv(VPP_data_input_path)
        # 转换成时间格式
        VPP_data['time'] = pd.to_datetime(VPP_data['time'])
        # 将某一列作为索引
        VPP_data = VPP_data.set_index('time')

        # 所有epi中的常量constant
        #Costants for all episodes:
        # 输入EV参数，调用Elvis,返回的EV抵达的分布
        self.elvis_config_file = elvis_config_file
        # '2022-01-01T00:00:00'
        self.start = simulation_param["start_date"]
        # '2023-01-01T00:00:00'
        self.end = simulation_param["end_date"]
        # 间隔15min   # 决策时间步长  '0:15:00'
        self.res = simulation_param["resolution"]

        # 家庭负荷数量  4
        num_households = simulation_param["num_households"]
        # 16
        self.solar_power = simulation_param["solar_power"]
        # 12
        self.wind_power = simulation_param["wind_power"]

        # 是个特斯拉的很多参数
        self.EV_types = simulation_param["EV_types"]

        # 4
        self.charging_stations_n = simulation_param["charging_stations_n"]
        # 'EV_arrivals(W)'=20、25...
        self.EVs_n = simulation_param["EVs_n"]
        # 25*52.2=1305
        self.EVs_n_max = simulation_param["EVs_n_max"]
        # hours = 23.99
        self.mean_park = simulation_param["mean_park"] #hours
        # 以小时为单位的标准差±1h
        self.std_deviation_park = simulation_param["std_deviation_park"] #std in hours
        # 换算成千瓦时  50
        self.EVs_mean_soc = simulation_param["EVs_mean_soc"]# %translated to kWh
        #   10
        self.EVs_std_deviation_soc = simulation_param["EVs_std_deviation_soc"]# %translated to kWh
        #   44
        self.EV_load_max = simulation_param["EV_load_max"]
        #self.EV_load_rated = simulation_param["EV_load_rated"]
        #   单个充电站最大充电功率  11
        self.charging_point_max_power = self.EV_load_max/self.charging_stations_n #kW
        #   单个充电站额定充电功率  3.7
        self.charging_point_rated_power = simulation_param["EV_load_rated"]/self.charging_stations_n #kW
        # 单个充电站最低充电功率  1
        self.charging_point_min_power = simulation_param["EV_load_min"]
        # 每多一户+1kW？   10+4=14
        self.houseRWload_max = simulation_param["houseRWload_max"] + (num_households * 1)
        # 0.13
        self.max_energy_price = simulation_param["av_max_energy_price"]
        # 100kWh
        main_battery_capacity = self.EV_types[0]["battery"]["capacity"]
        self.battery_max_limit = main_battery_capacity - ((main_battery_capacity/100)*0.01) #99.99kWh
        self.battery_min_limit = (main_battery_capacity/100)*0.01 #0.01kWh
        #  低于20%不能放电，不可放电
        self.DISCHARGE_threshold = 20 #percentage of battery below with the EV can't be discharged
        # 低于10%不能放电且必须充电. 电池百分比以下的电动车不能放电，保持闲置（必须充电）。
        self.IDLE_DISCHARGE_threshold = 10 #percentage of battery below with the EV can't be discharged, kept idle (must be charged)

        # 调用Elvis,车的抵达分布和开始结束调度周期
        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)

        # EVs初始化仿真, 将在每一epi更新
        #ELVIS Initial simulation
        #To be updated each episode:
        # 事件只能从独立的每周到达分布中取样，即每周GMM（高斯混合模型）
        # 我猜是生成一些车来充电事件
        self.charging_events = elvis_realisation.charging_events
        print(self.charging_events[0], '\n', '...', '\n', self.charging_events[-1], '\n')
        # Charging event: 1, Arrival time: 2022-01-01 06:45:00, Parking_time: 24, Leaving_time: 2022-01-02 06:45:00, SOC: 0.5806366821500033, SOC target: 1.0, Connected car: Tesla, Model S
        #  ...
        #  Charging event: 1304, Arrival time: 2022-12-31 12:45:00, Parking_time: 24, Leaving_time: 2023-01-01 12:45:00, SOC: 0.46525462628746506, SOC target: 1.0, Connected car: Tesla, Model S

        # reset中用到了
        self.current_charging_events = []
        # 共1304个事件
        self.simul_charging_events_n = len(self.charging_events)
        # 创建序列15分钟一个点，从开始，结束日期和决议的模拟期间的所有单独的时间序列。 create_time_steps：与elvis有关并在elvis内部多次使用的方法
        self.elvis_time_serie = create_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution)

        # 用于临时将stdout重定向到另一个文件的上下文管理器
        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            # simulate(车的抵达分布和开始结束调度周期),要模拟的场景
            result = simulate(elvis_realisation)
        
        # 用时间序列作为索引,生成的汽车序列与VPP住宅的数据对应上
        VPP_data["time"] = self.elvis_time_serie
        VPP_data = VPP_data.set_index("time")
        # 检查环境设定功率是否需要调整???咋调的没看懂
        #Check if environment setting power needs to be rescaled
        if self.solar_power!=16 or self.wind_power!=12 or num_households!=4:
            VPP_data["solar_power"] = VPP_data["solar_power"]/16 * self.solar_power
            VPP_data["wind_power"] = VPP_data["wind_power"]/12 * self.wind_power
            VPP_data["household_power"] = VPP_data["household_power"]/4 * num_households
            VPP_data["House&RW_load"] = VPP_data["household_power"] - VPP_data["solar_power"] - VPP_data["wind_power"]
        # House&RW_load=本地负载-风光之后还需要的 
        # RW_power是新能源=风+光发电
        VPP_data["RW_power"] = VPP_data["solar_power"] + VPP_data["wind_power"]
        # 生成场景.汇总负荷情况(生成的步数)，即一系列的负荷数据点
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        # 15分钟的电价  需要/4
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"] / 4
        self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh
        # 除去风光后的住宅负荷
        # VPP_loads ：时间、住宅还需负荷、住宅负荷、光、风共5列数据
        self.houseRW_load = VPP_data["House&RW_load"].values

        self.VPP_loads = pd.DataFrame({'time':self.elvis_time_serie, "House&RW_load":self.houseRW_load, "household_power":VPP_data["household_power"].values, "solar_power":VPP_data["solar_power"].values, "wind_power":VPP_data["wind_power"].values})
        self.VPP_loads = self.VPP_loads.set_index("time")
        
        # 29045 住宅总负荷（不考虑风光）
        self.household_consume = VPP_data["household_power"].sum()/4 #kWh
        # 50260.26
        self.RW_energy = VPP_data["RW_power"].sum()/4 #kWh
        # 除去风光后的负荷需求  35041组数据，注365*24*4=35040
        HRW_array = np.array(self.houseRW_load)
        # 除去风光后的负荷需求总数有正负    -21214.63
        self.sum_HRW_power = np.sum((HRW_array)/4) #kWh
        # 风光不够住宅用电，又从电网买的>0求和4947kWh
        self.HRW_overenergy = HRW_array[HRW_array>0].sum()/4 #kWh (Grid energy used)
        # 风光够负荷用后，<0的多的转化为EV的闲置能源-261261kWh
        self.HRW_underenergy = HRW_array[HRW_array<0].sum()/4 #kWh (RE-to-vehicles unused energy)
        # 自消纳=住宅总功耗不考虑风光-从电网买的电    24098 
        self.self_consumption = self.household_consume - self.HRW_overenergy
        # 自消纳率=住宅消纳的风光/风光总产出  47.9
        self.selfc_rate = (self.self_consumption / self.RW_energy) * 100
        # 自给自足率=住宅消纳的风光/住宅总负荷（不考虑风光）  82.9
        self.autarky_rate = (self.self_consumption / self.household_consume) * 100
        
        # 除去风光后买卖电的钱序列，注意有正负
        dataset_cost_array = HRW_array * np.array(self.prices_serie)/4 
        # 除去风光后买卖电的钱总数  -489.7€
        self.cost_HRW_power = dataset_cost_array.sum() #€
        #  除去风光后只买电的钱总数，为正  233.11€
        self.overcost_HRW_power = dataset_cost_array[dataset_cost_array>0].sum() #€
        # 预计充完后EV的平均剩余电量 = 50kwh + (-供EV的闲置能源/EV数量) = 70.06 kwh
        self.exp_ev_en_left = self.EVs_mean_soc + (-self.HRW_underenergy/self.simul_charging_events_n)
        #ELVIS
        # 住户和EV去除风光的总负荷
        load_array = np.array(VPP_data["total_load"].values)
        # 总成本=住户和EV去除风光的总负荷total_load*EUR/kWh电价/4 15分钟电价
        cost_array = np.array(VPP_data["total_cost"].values)
        # 住户和EV去除风光的总负荷的均值 3.12kW
        self.av_Elvis_total_load = np.mean(load_array) #kW
         # 住户和EV去除风光的总负荷的标准差9.2kW
        self.std_Elvis_total_load = np.std(load_array) #kW
         # 住户和EV去除风光的总能耗=total_load求和/4 27351kWh
        self.sum_Elvis_total_load = load_array.sum()/4 #kWh
        # EV+住户去除风光买电 43704kwh
        self.Elvis_overconsume = load_array[load_array>0].sum()/4 #kWh
        # EV+住户去除风光剩余的 16352kwh 
        self.Elvis_underconsume = -load_array[load_array<0].sum()/4 #kWh
        # 1064€ 总消费有正负 = total_cost求和    之前已经除过4了
        self.Elvis_total_cost = np.sum(cost_array) #€
        # 只买电的总消费= total_cost > 0求和  # 1535.9€
        self.Elvis_overcost = cost_array[cost_array > 0].sum()
        # 初始打印输出  四舍五入到给定的小数2位精度   
        #Init print out
        # 1除去风光后的住户负荷需求总数有正负 无EV 
        # 2无EV买电网的电     总需求:self.household_consume
        # 3无EV自给自足率=住宅消纳的风光/住宅总功耗 
        # 4无EV多的可转化为EV的闲置能源 
        # 5无EV自消纳率=住宅消纳的风光/风光总产出 
        # 6无EV除去风光后买卖电的钱总数 
        # 7无EV除去风光后只买电的钱总数只为正
        print("-DATASET: House&RW_energy_sum=kWh ", round(self.sum_HRW_power,2),
                f",\nGrid_used_en(grid-import)={round(self.HRW_overenergy,2)}kWh",
                f", \nautarky-rate={round(self.autarky_rate,1)}",
                f", \nRE-to-vehicle_unused_en(grid-export)={round(self.HRW_underenergy,2)}kWh",
                f", \nself-consump.rate={round(self.selfc_rate,1)}",
                ", \nTotal_selling_cost=€ ", round(self.cost_HRW_power,2),
                ", \nGrid_cost=€ ", round(self.overcost_HRW_power,2),"\n")
        # 设定的平均SoC容量50kwh 
        # 1住户和EV去除风光的总能耗=total_load求和/4 
        # 2EV+住户去除风光买电 
        # 4EV+住户去除风光剩余的 即total_load<0的部分取正
        # 6总消费有正负=total_cost求和之前已除4 
        # 7只买电的总消费=total_cost>0求和
        # 共1304个事件 
        # 买电,加入EV后未使用的 ,买电消费,预计充完后EV的平均剩余电量=50kwh+(-供EV的闲置能源/EV数量) 
        print("\n\n- ELVIS.Simulation (Av.EV_SOC= ", self.EVs_mean_soc, "%):",
            "\nSum_Energy=kWh ", round(self.sum_Elvis_total_load,2),
            ",\nGrid_used_en=kWh ", round(self.Elvis_overconsume,2),
            ", \nRE-to-vehicle_unused_en=kWh ", round(self.Elvis_underconsume,2),
            ", \nTotal_selling_cost=€ ", round(self.Elvis_total_cost,2),
            ", \nGrid_cost=€ ", round(self.Elvis_overcost,2), 
            ", \nCharging_events= ", self.simul_charging_events_n,
            "\n- Exp.VPP_goals: Grid_used_en=kWh 0, RE-to-vehicle_unused_en=kWh 0, Grid_cost=€ 0",
            ", Av.EV_en_left=kWh ",round(self.exp_ev_en_left,2),"\n")

        #设置 VPP 会话长度 35041  
        #Set VPP session length
        self.tot_simulation_len = len(self.elvis_time_serie)
        self.vpp_length = self.tot_simulation_len
        #empty list init
        # ,,,,奖励历史
        self.energy_resources, self.avail_EVs_id, self.ev_power, self.charging_ev_power, self.discharging_ev_power, self.overcost, self.total_cost, self.total_load, self.reward_hist = ([],[],[],[],[],[],[],[],[])
        # 负荷功率上限kw
        self.max_total_load = self.EV_load_max + self.houseRWload_max
        # 每15分钟的最大消费
        self.max_cost = self.max_total_load * self.max_energy_price /4
        # 设置奖励插值功能的函数.考虑分段，如使用EV能充到的均值70%作为最高奖励  
        #Setting reward functions
        self.set_reward_func()

        self.VPP_data = VPP_data
        self.VPP_actions, self.action_truth_list, self.EVs_energy_at_leaving, self.EVs_energy_at_arrival = ([],[],[],[])
        #self.lstm_states_list = []
        self.av_EV_energy_left, self.std_EV_energy_left, self.sim_total_load, self.sim_av_total_load, self.sim_std_total_load, self.overconsumed_en, self.underconsumed_en, self.sim_total_cost, self.sim_overcost = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cumulative_reward, self.load_t_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward, self.EVs_energy_reward, self.AV_EVs_energy_reward = [0, 0, 0, 0, 0, 0, 0]
        #  为第一步[0]初始化状态空间参数 
        #Initializing state space parameters for the first step [0]
        # 充电站的可用能量（连接的电动车）：充电站，4个0
        Init_space_Available_energy_sources = np.zeros(self.charging_stations_n, dtype=np.float32)
        #Init_space_Available_EVs_id = np.zeros(self.charging_stations_n, dtype=np.int32)
        # EV功率、去除风光还需的总负荷 设为0.
        Init_space_ev_power = np.zeros(1,dtype=np.float32)
        Init_space_total_load = np.zeros(1,dtype=np.float32)
        #Init_space_total_cost = np.zeros(1,dtype=np.float32)
        # 无EV去除风光还需的总负荷[0] 取第一个点
        # 看第一行的值即可，最大最小是pandas统计的
        Init_space_total_load[0] = self.houseRW_load[0]
        #Init_space_total_cost[0] = self.houseRW_load[0] * self.prices_serie[0]/4

        self.Init_space = {
            # 电动车负载范围，单位为千瓦，取决于模拟的基础设施 
            'ev_power': Init_space_ev_power, #EV load range in kW, depends on the infrastructure of the simulation
            # 总负荷范围，以千瓦为单位，取决于模拟的家庭和电动车基础设施 
            'total_load': Init_space_total_load, #Total load range in kW, depends on the household and EV infrastructure of the simulation
            #'total_cost': Init_space_total_cost, #Total cost (EUR) per timestep (15 min) [DELETED]
            # 充电站的可用能量（连接的电动车）
            'Available_energy_sources': Init_space_Available_energy_sources, #Available energy from Charging stations (EVs connected)
            # 时间步数t时充电站的可用车辆ID 
            #'Available_evs_id': Init_space_Available_EVs_id #Available vehicles IDs from charging stations at timestep t [DELETED]
            }
        self.space_0 = self.Init_space
        
        # 用于绘制电池电量图 
        #For plotting battery levels
        self.VPP_energies = Init_space_Available_energy_sources
        # Define constants for Action space options
        # 闲置
        self.IDLE = 0
        # 充
        self.CHARGE = 1
        # 放
        self.DISCHARGE = 2
        # 额定功率
        #self.VPP_rated_pwr = 3.7 #kW
        # 可采取的行动
        self.possible_actions = 3
        # 动作空间的定义[3, 3, 3, 3] 
        #Action space definition
        self.action_space = MultiDiscrete( self.possible_actions * np.ones(self.charging_stations_n))
        #用于动作遮蔽的动作集定义(无效动作惩罚、无效动作遮蔽、其他处理无效动作https://zhuanlan.zhihu.com/p/358738280)
        #Actions set definition for action masking
        dims = [self.possible_actions]*self.charging_stations_n
        # [0, 1, 2, 3, 4, 5...11],0~3分别表示四充电站闲置 4~7四个充电站充,8~11表示四个充电站放电
        self.actions_set = np.arange(sum(dims))
        # [11个True]
        self.invalid_actions_t = np.ones(len(self.actions_set), dtype=bool)
        self.lstm_state = None

        spaces = {
            # 电动车负载范围，单位为千瓦，取决于模拟的基础设施[-44], [44], (1,) 
            'ev_power': Box(low=-(self.EV_load_max), high=(self.EV_load_max), shape=(1,), dtype=np.float32), #EV load range in kW, depends on the infrastructure of the simulation
            #总负荷范围，以千瓦为单位，取决于模拟的家庭和电动车基础设施[-58], [58], (1,)  
            'total_load': Box(low=-(self.max_total_load) , high= (self.max_total_load), shape=(1,), dtype=np.float32), #Total load range in kW, depends on the household and EV infrastructure of the simulation
            # 每个时间段（15分钟）的总成本（欧元）[-1.885], [1.885], (1,)
            #'total_cost': Box(low=-(self.max_cost), high=(self.max_cost), shape=(1,), dtype=np.float32),#Total cost (EUR) per timestep (15 min) [DELETED]
            # 4维[0, 0, 0, 0], [100, 100, 100, 100], (4,) 
            'Available_energy_sources': Box(low=0.0, high=100, shape=(self.charging_stations_n,), dtype=np.float32),
            #'Available_evs_id': Box(low=0, high=(np.iinfo(np.int32).max)-1, shape=(self.charging_stations_n,), dtype=np.int32) [DELETED]
            }

        dict_space = Dict(spaces)
        self.observation_space = dict_space
        #设定起始条件。
        #Set starting cond.
        # 即{ev_power:[0], total_load:[1.71759], Available_energy_sources:[0., 0., 0., 0.]}
        self.state = self.Init_space
        self.done = False

    def set_reward_func(self):
        """
        设置奖励插值功能的函数.考虑分段，如使用EV能充到的均值70%作为最高奖励
        Function to set the reward interpolating functions
        """
        #Step rewards
        #step EV energies -> [0, 50kWh, (75-80kWh), 100kWh]
        self.battery_percentage =[0, self.EVs_mean_soc, 90, 100]
        self.EVs_energy_reward_range = np.array([-300, -40, +150, 50])

        # 根据数据集最大负载规范化负载范围 
        #step Load reward #Normalizing Load range according to datset max load (self.max_total_load = 40 kW)
        load_array = np.array([-80, -30, -15, -4, -1, 0, 0.5, 3.5, 15, 30, 80]) #kW
        self.load_range = load_array
        #self.load_range = (self.max_total_load/100)*np.array([-100, -35, -15, -3, -1.5, 0.1, 3, 15, 35, 100])
        #[-40kW, -12, -6kW, -2kW, -0.4kW, 0, 0.04kW, 2kW, 20kW, 40kW]
        self.load_reward_range = np.array([-50, -30, -15, -5, 0, 15, 0, -5, -20, -40, -80])

        # 最终奖励
        # 预计充完后EV的平均剩余电量的奖励
        #FINAL REWARDS
        #Av_EV_energy_left reward
        #self.av_energy_left_range = [0, 50, 60, 90, 100]
        self.av_energy_left_range = [0, self.EVs_mean_soc, self.exp_ev_en_left, 100]
        self.av_energy_reward_range = np.array([-50000, -10000, 30000, 10000])

        #Average load #Normalizing av.load according to av_Elvis_load = 6.85 kW
        # EV+住户去除风光买电总量/100*[0~200],买的越多奖励越低
        self.overconsume_range = (self.Elvis_overconsume/100)*np.array([0, 2, 100, 200]) #Elvis over-consume=kWh  43221.9
        #av_load_label = ["0 kWh", "800 kWh", "8000 kWh", "40000 kWh"]
        self.overconsume_reward_range = np.array([1000, 0, -2000, -5000])*20

        #Stand_dev load #Normalizing std.load according to std_Elvis_load = 11.96
        self.underconsume_range = (self.Elvis_underconsume/100)*np.array([0, 20, 100, 200]) #Elvis under-consume=kWh  14842.72
        #std_load_label = ["0 kWh", "2000 kWh","15000 kWh", "30000 kWh"]
        self.underconsume_reward_range = np.array([1000, 0, -3000, -5000])*5

        #total COST #Normalizing total cost according to Elvis_overcost = 2115€
        self.overcost_range = (self.Elvis_overcost/100)*np.array([0, 10, 30, 100, 150]) #Elvis overcost=€  1514.69 
        #cost_label = ["0€", "200€", "800€", "1200€", "2000€"]
        self.overcost_reward_range = np.array([2000, 100, 0, -1000, -2000])*10


    def eval_reward(self, reward, step, new_ev_departures):
        """
        在仿真的每个时间段评估agent的奖励
        Function to evaluate the agent reward at each timestep of the simulation
        """
        #加载当前step的状态变量 
        #Load step state variables
        total_load_t = self.total_load[step]
        # 充电站每个EV的剩余能量  
        #energy_resources = self.energy_resources[step] #[DELETED]
    
        # 奖励：充电站可提供的能量   EVs
        #EVs reward: energies available at charging stations
        EVs_energy_reward_t = 0
        # 在每个时间段对电动车的可用能量进行奖励。对Agent来说，学习策略是很混乱的 
        #Reward at each timestep for EVs available energy--> confusing for the Agent to learn the policy [DELETED]
        """ for n in range(self.charging_stations_n):
            检查电动车是否连接并评估在n充电站的奖励
            #1. Check if Evs connected and evaluate reward at station n
            if energy_resources[n] > 0:
                # 插补 插值
                EVs_energy_reward_t += np.interp(energy_resources[n], self.battery_percentage, self.EVs_energy_reward_range)
        self.EVs_energy_reward += EVs_energy_reward_t """

        EVs_energy_leaving_reward_t = 0
        # 对离开车站时留在车上的能量进行奖励（它强调好/坏的行为）。
        #Apply reward on energy left on vehicle WHEN leaving the station (it accentuate good/bad behaviour)
        # 0,1,2,...        ???[-1-j]
        for j in range(new_ev_departures):
            # 如当前t有2个车走了,都记录到以前的仅离开记录表(如step=1501,表里有67个记录),从最后一个开始、倒数第二算刚刚离开的这两车的奖励,  EVs_energy_at_leaving就是Energy_sources_t.append一个个加上的(所有step离开的都在),第t步的可用每个充电站EV剩余能量, 不涉及上一步都是这一步的临走记录一下,
            energy_left = self.EVs_energy_at_leaving[-1-j]
            # 这一step离开了几个车,就累加几个奖励
            EVs_energy_leaving_reward_t += np.interp(energy_left, self.battery_percentage, self.EVs_energy_reward_range)
        self.EVs_energy_reward += EVs_energy_leaving_reward_t
        # 执行情况的奖励记录到上一step的hist历史中
        self.EVs_reward_hist[step-1] = EVs_energy_leaving_reward_t

        # 每个时间step负载奖励,越接近0奖励越大   
        #Load reward for each timestep
        # total_load=House&RW_load+ev_power=住宅去除风光还需负荷功率(还需功率时为负载为正)+EV总功率(充电时为负载为正),EV够用电网不买电为0.插值,一维线性内插
        load_reward_t = np.interp(total_load_t, self.load_range, self.load_reward_range)
        # 负载累积奖励 越靠近0越高. done的时候重新求和算了下 self.load_t_reward = np.sum(self.load_reward_hist)
        self.load_t_reward += load_reward_t
        self.load_reward_hist[step-1] = load_reward_t

        # 3个中弃用了第1个
        reward += (EVs_energy_reward_t + EVs_energy_leaving_reward_t + load_reward_t)
        return reward
    
    def eval_final_reward(self, reward):
        """
        在仿真结束时评估最终的agent奖励
        Function to evaluate the final agent reward at the end of the simulation
        """
        #电动汽车的能源奖励。评估电动车离开时的平均剩余能量的奖励如70%最高 
        #EVs ENERGY reward: Evaluating reward for average energy left in EV leaving
        AV_EVs_energy_reward = np.interp(self.av_EV_energy_left,
                    self.av_energy_left_range,
                    self.av_energy_reward_range)
        self.AV_EVs_energy_reward += AV_EVs_energy_reward

        # 负载奖励 
        #LOAD reward:
        # 买电越多越惩罚  
        #Overconsumed energy reward
        final_overconsume_reward = np.interp(self.overconsumed_en, self.overconsume_range, self.overconsume_reward_range)
        self.overconsume_reward += final_overconsume_reward

        #未消纳的风光(total_load中<0的部分求和/4)越多越惩罚
        #Underconsumed energy reward
        final_underconsume_reward = np.interp(self.underconsumed_en, self.underconsume_range, self.underconsume_reward_range)
        self.underconsume_reward += final_underconsume_reward

        #只买电的总消费求和=np.sum(self.overcost)越多越惩罚
        #OverCOST reward:
        final_overcost_reward = np.interp(self.sim_overcost, self.overcost_range, self.overcost_reward_range)
        self.overcost_reward += final_overcost_reward

        reward += (AV_EVs_energy_reward + final_overconsume_reward + final_underconsume_reward + final_overcost_reward)
        return reward
    
    def action_masks(self):
        """
        根据step观察，评估agent在每个时间点不应该采取的 "无效 "行动。被Maskable PPO算法用于训练和预测，被所有的算法仿真用于一般行为控制
        Function to evaluate the "invalid" actions the agent should not take at each timestep depending on the step observation
        (used by the Maskable PPO algorithm for training and prediction and by all the algorithms simulation for general behaviour control)
        """
        # 闲置
        #self.IDLE = 0
        # 充
        #self.CHARGE = 1
        # 放
        #self.DISCHARGE = 2
        # 这两个有啥差距,得到第几步,35041
        step = self.tot_simulation_len - self.vpp_length    
        #loding step variables
        # 第t步每个充电站的EV的ID [0,0,0,1305]
        Evs_id_t = self.avail_EVs_id[step]
        # 可用EV充电站数
        EVs_available = 0
        # 统计4充电站的可用EV,有id就+1,最大为4
        for n in range(self.charging_stations_n):
            if Evs_id_t[n] > 0: EVs_available+=1
        # 第t步的可用每个充电站剩余能量
        Energy_sources_t = self.energy_resources[step]
        # 下一步的住宅还需负荷
        houseRW_load = self.houseRW_load[step+1]
        # 有问题的动作
        invalid_actions = []

        # 4个充电站依次
        for n in range(self.charging_stations_n):
            #  如果没车辆在n站
            if Evs_id_t[n] == 0:#if vehicle not present at station n:
                # 充放无效[1~3)即1,2,然后外层充电站n个再迭代   
                for i in range(1, self.possible_actions): #CHARGE,DISCHARGE invalid
                    # 第0~3充电站充无效 0+1*4=4  0+2*4=8 1+1*4=5 1+2*4=9 2+1*4=6 2+2*4=10 3+1*4=7 3+2*4=11 即[4,5,6,7,8,9,10,11]其中8,9,10,11代表第0~3站放电无效,其他代表充电无效
                    invalid_actions.append(n + i*self.charging_stations_n)
            else:
                #如果车辆在n站出现 
                #IF vehicle present at station n:
                # 低于10%不能放电且必须充电
                if Energy_sources_t[n] <= self.IDLE_DISCHARGE_threshold:
                    #低于40%电量则闲置、放电无效  
                    #IDLE,DISCHARGE invalid if battery below 40%
                    # i=0,2即不能闲置和放电     步长为2，代表取出规则是“取一个元素跳过一个元素再接着取
                    for i in range(self.IDLE, self.possible_actions, 2):
                        # 若n=0,则0+0*4=0, 0+2*4=8  若n=1则 1+0*4=1,1+2*4=9  解释:4代表第0充电站不能闲置,8代表第0充电站不能放电,1代表第1充电站不能闲置,9代表第1充电站不能放电
                        invalid_actions.append(n + i*self.charging_stations_n)

                # EV剩余在10%～20%?  上面写的低于20%不能放电，可不充电
                elif self.IDLE_DISCHARGE_threshold < Energy_sources_t[n] <= self.DISCHARGE_threshold:
                    #电量低于55%，放电无效  
                    #DISCHARGE invalid if battery below 55%
                    # 放电=2不能用,n=0,则0+2*4=8,表示第0个充电站不能放
                    invalid_actions.append(n + self.DISCHARGE*self.charging_stations_n)

                elif Energy_sources_t[n] > 91:
                    #如果电池超过91%，CHARGE无效 
                    #CHARGE invalid if battery over 91%
                    invalid_actions.append(n + self.CHARGE*self.charging_stations_n)

                # 如果负载除去风光还需能源为正
                if houseRW_load > 0: #if load positive
                    # 充电无效 
                    invalid_actions.append(n + self.CHARGE*self.charging_stations_n) #CHARGE invalid

                    # 如果只有一辆车可用,闲置无效,放电有效  
                    if EVs_available == 1: #if only vehicle available:
                        invalid_actions.append(n) #IDLE invalid
                        # 如果放电无效
                        if (n + self.DISCHARGE*self.charging_stations_n) in invalid_actions: #if DISCHARGE was invalid
                            # 让放电有效 
                            invalid_actions.remove(n + self.DISCHARGE*self.charging_stations_n) #DISCHARGE valid
                    # 如果不是唯一的车辆,让最大电量的放电有效,闲置无效 
                    elif EVs_available > 1: #if not the only vehicle available:
                        # 电池百分比>0则拿出作为列表,第n个充电站都>=列表的值,即有最大电量,则不闲置 
                        if all( Energy_sources_t[n] >= x for x in [bat_perc for bat_perc in Energy_sources_t if bat_perc > 0]):#if vehicle with most charge
                            # 闲置无效
                            invalid_actions.append(n) #IDLE invalid
                            # 如果放电无效
                            if (n + self.DISCHARGE*self.charging_stations_n) in invalid_actions: #if DISCHARGE is invalid
                                # 让放电有效
                                invalid_actions.remove(n + self.DISCHARGE*self.charging_stations_n) #DISCHARGE valid


                # 如果负载-风光为负,即有闲置
                elif houseRW_load < 0: #if load is negative
                    # 放电无效
                    invalid_actions.append(n + self.DISCHARGE*self.charging_stations_n) #Discharge invalid
                    #for i in range(self.IDLE, self.possible_actions, 2):
                    #    invalid_actions.append(n + i*self.charging_stations_n) #IDLE,DISCHARGE invalid
                    if EVs_available == 1: #if only vehicle available:
                        # 只剩一辆则闲置无效 
                        invalid_actions.append(n) #IDLE invalid
                        # 如果充电无效
                        if (n + self.CHARGE*self.charging_stations_n) in invalid_actions: #if CHARGE was invalid
                            # 让充电有效
                            invalid_actions.remove(n + self.CHARGE*self.charging_stations_n) #CHARGE valid
                    # 可用EV>1
                    elif EVs_available > 1: #if not the only vehicle available:
                        # 如果车辆的充电量最少
                        if all( Energy_sources_t[n] <= x for x in [bat_perc for bat_perc in Energy_sources_t if bat_perc > 0]):#if vehicle with least charge
                            # 闲置无效
                            invalid_actions.append(n) #IDLE invalid
                            # 如果充电无效
                            if (n + self.CHARGE*self.charging_stations_n) in invalid_actions: #if CHARGE is invalid
                                # 让充电有效
                                invalid_actions.remove(n + self.CHARGE*self.charging_stations_n) #CHARGE valid

        invalid_actions = [*set(invalid_actions)]
        # 表示0~3充电站只能闲置, self.actions_set=[0, 1, 2, 3, 4, 5...11], [T,T,T,T,F,F,F,F,F,F,F]
        self.invalid_actions_t = [action not in invalid_actions for action in self.actions_set]
        return self.invalid_actions_t
    
    def apply_action_on_energy_source(self, step, Energy_sources_t_1, action, total_ev_power_t, ch_station_ideal_pwr):
        """
        以下都针对单个充电桩(即EV) 1.根据step函数中的风光负荷剩下的能源计算的充电站理想功率,考虑SoC确定自适应功率变量("selected_power"kW*15min=kWh)  2.根据1,执行动作(考虑特殊情况,10%,20%,充超,放超),得到ev_power_t和Energy_sources_t、total_ev_power_t
        将agent选择的动作（IDLE闲置,CHARGE充,DISCHARGE放）应用于有电动汽车存在的所选定的充电站。自适应功率选择根据agent选择的行动计算功率以获得总的零负荷。
        Function to apply the agent's chosen action (IDLE,CHARGE,DISCHARGE) to the selected charging station with an EV present.
        the Adaptive power selection calculates the power to get a total zero-load according to the agent's actions chosen.
        """
        #自适应功率选择"selected_power"     
        #Adaptive power selection
        # 计划充电:若动作也是充或者上一次的SoC<10%就按理想功率充,否则(动作是放电且容量>10%)仍按最小充电指标充
        if ch_station_ideal_pwr > 0:
            # or 充电站上一次容量<10%,则充
            if action == self.CHARGE or (Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold):
                selected_power = ch_station_ideal_pwr
            else: selected_power = self.charging_point_min_power
        # 计划放电:上一次容量<10%,则用额定充;否则>10%,若动作为放电,则功率为计划理想功率(取正);>10%,动作为充电,按最小功率1充.
        elif ch_station_ideal_pwr < 0:
            if (Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold):
                selected_power = self.charging_point_rated_power
             # 否则,若动作为放电,则功率为计划理想功率(取正)
            elif action == self.DISCHARGE:
                selected_power = -ch_station_ideal_pwr
            # 否则最小功率1
            else: selected_power = self.charging_point_min_power
        # 计划理想功率=0,则按额定3.7充
        else: selected_power = self.charging_point_rated_power
        
        battery_max_limit = self.battery_max_limit #99.9 kWh
        battery_min_limit = self.battery_min_limit #0.1 kWh

        # 根据动作决定EV的充放功率(有正负),并计算总的EV功率total_ev_power_t
        ev_power_t = 0
        # 对上一步的能源状态执行动作(未看完),充超(动作充、动作闲置下强制充、动作放电下强制充)了上限就改功率到正好充满,放超(动作放电)了下限就改功率正好为0
        #APPLY ACTION on previous energy state:
        if action == self.CHARGE:
            # 第t步的EV能量
            Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
            ev_power_t += selected_power
            if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                # 加上充的已经超了电池容量咋办,超的部分(kWh)再除0.25就是超出的功率,则正好充满的ev_power_t功率=原功率-超出的功率
                ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                # 重设为正好充满
                Energy_sources_t = battery_max_limit
        elif action == self.IDLE:
            # 如果特殊情况,能量低于空闲-放电阈值（10%）--> 强制充电不能闲置
            if Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold: #if energy below the Idle-Discharge threshold (10%) --> CHARGE
                # 计算充15分钟后下一步的能源,存储能源  
                Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
                # EV功率累加
                ev_power_t += selected_power
                # 由10%充满了达到最大容量（99.99千瓦时）,超出部分容量/4取负再加到EV功率上,限制容量为最大值
                if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                    ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                    Energy_sources_t = battery_max_limit
            #elif Energy_sources_t_1 > self.IDLE_DISCHARGE_threshold: #if energy above the Idle-Discharge threshold (10%) --> IDLE
            # 执行动作(闲置),保持能量不变
            else: Energy_sources_t = Energy_sources_t_1 #keep energy constant
        elif action == self.DISCHARGE:
            # 如果特殊情况,能量低于空闲-放电阈值（10%）-->强制充电不能放电
            if Energy_sources_t_1 <= self.IDLE_DISCHARGE_threshold: #if energy below the Idle-Discharge threshold (10%) --> CHARGE
                Energy_sources_t = Energy_sources_t_1 + (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh STORING ENERGY
                ev_power_t += selected_power
                # 由10%以下充超了最大容量（99.99千瓦时）,超出部分容量/4累减到EV功率上???,限制容量为最大值
                if Energy_sources_t > battery_max_limit: #Reached max capacity (kWh)
                    ev_power_t -= (Energy_sources_t - battery_max_limit)/0.25
                    Energy_sources_t = battery_max_limit
            # 如果正常情况,能量超过放电阈值（25%）-->执行放电, 容量减少,EV功率累减
            elif Energy_sources_t_1 > self.DISCHARGE_threshold: #if energy above the Discharge threshold (25%) --> DISCHARGE
                Energy_sources_t = Energy_sources_t_1 - (selected_power * 0.25) #5 kW * 15 min = 1.25 kWh PUSHING ENERGY
                ev_power_t -= selected_power
                # 由20%以上放超了最小容量（0.01千瓦时）,超出部分容量(负)/4就是超的功率,加到原EV功率上正好能放到最小,限制容量为最小值
                if Energy_sources_t < battery_min_limit: #Reached min capacity (kWh)
                    ev_power_t += (battery_min_limit - Energy_sources_t)/0.25
                    # 重设为最小值
                    Energy_sources_t = battery_min_limit
            # 如果特殊情况,能量低于放电阈值(25%)-----强制为空闲状态 
            #elif Energy_sources_t_1 <= self.DISCHARGE_threshold: #if energy below the Discharge threshold (25%) --> IDLE
            # 10~20%保持能量不变
            else: Energy_sources_t = Energy_sources_t_1 #keep energy constant
        # 动作空间的值不对
        else:
            raise ValueError("Received invalid action={} which is not part of the action space".format(action))

        #充电/放电功率累计序列更新,充电为正  
        #Charge/discharge power series update
        if ev_power_t < 0:
            self.discharging_ev_power[step] += ev_power_t
        elif ev_power_t > 0:
            self.charging_ev_power[step] += ev_power_t
        # 累计ev功率, 当前step的EV需要的总功率(除4就是总能耗),充电为正放电为负
        total_ev_power_t += ev_power_t

        return Energy_sources_t, total_ev_power_t


    def step(self, action):
        """
        在仿真过程每一步更新环境：
        0. 行动分析和自适应功率评估(由风、光、用户负荷、充电站EV剩余SoC、充电站动作 决定出剩余的能源并计算该以怎样的理想功率进行充电(ch_station_ideal_pwr未除4),后面selected_power也考虑SoC即是否10%和20%闲置区间)
        1. 检查离开充电站的电动车（在离开前对这些电动车应用行动，然后更新目前的电动车ID阵列和当前的充电事件列表）
        2. 对EVs ID阵列中存在电动车的每个充电站实施行动 
        3. 检查新的电动汽车到达充电站（更新当前电动汽车ID阵列、模拟充电事件列表和当前正在充电的事件列表）。
        4. 更新VPP环境状态和价值
        5. 最后，仿真结束后，评估最终奖励、数据结构和性能。
        Function to update the environment every step along the simulation. Divided in 5 main sections:
        - 0. Actions analysis and Adaptive power evaluation
        - 1. EVs leaving the charging stations check (apply action on those before leaving, then update the present EVs ID array and current_charging_events list)
        - 2. Apply action to each station with an EVs present in the EVs ID array
        - 3. New EVs arrival at the charging stations check (update the present EVs ID array, the simulation charging_events list and the current_charging_events list)
        - 4. Update the VPP environment States and Values
        Final_section: Evaluate final reward, data structures and performance when the simulation is over.
        """
        #Reduce VPP session length by 1 step [Episode init and reset do as step 0, first step = 1]
        self.vpp_length -= 1
        
        #evaluate step
        # step = 1,  vpp_length上面step每执行一次就-1   其他地方self.vpp_length = self.tot_simulation_len
        step = self.tot_simulation_len - self.vpp_length
        # 2022,1.1,0:15,
        time_step = self.elvis_time_serie[step]
    
        #加载step变量  
        #loding step variables
        # 第t个step的4个充电站EV数量
        Evs_id_t = self.avail_EVs_id[step]
        # 第t个step的4个充电站EV能量
        Energy_sources_t = self.energy_resources[step]
        # 从第2个开始赋值
        houseRW_load_t = self.houseRW_load[step]
        # 没用到? l代表last上次  上一step的4个充电站EV数量[0,0,0,0]   
        Evs_id_t_1 = self.avail_EVs_id[step-1] #not used
        Energy_sources_t_1 = self.energy_resources[step-1]

        #奖励初始化 
        #Reward Initialization
        reward = 0
        #用于中间计算的变量初始化  
        #Variables inititalization for calculations
        # 总EV功率
        total_ev_power_t = 0
        # [T,T,T,T]
        action_truth_array = np.ones(self.charging_stations_n, dtype = bool)
        # 新EV离开数
        new_ev_departures = 0
        # 要移除的充电事件
        charging_events_to_remove = []

        #0行动分析和自适应功率评估（见上面的解释. 即. 如果没有选择正确的行动，充电站理想充电功率 = 0）  
        #SECTION O. Actions analysis and Adaptive power evaluation (ch_station_ideal_pwr = 0 if no correct actions selected)
        # 充电动作,强制充电动作,放电动作,充电站理想功率
        charge_actions, forced_charge_actions, discharge_actions, ch_station_ideal_pwr = (0,0,0,0)
        for n in range(self.charging_stations_n):
            #if Evs_id_t_1[n] > 0: EVs_available+=1
            # 随机的action=[0,1,2,1]中对应充电站有充电行为即值为1,charge_actions就+1
            if action[n] == self.CHARGE: charge_actions+= 1
            # 如果有放电的行为,即值=2
            elif action[n] == self.DISCHARGE:
                # 上一step的可用能>20%电量则放电动作+1
                if Energy_sources_t_1[n] > self.DISCHARGE_threshold: discharge_actions+= 1
                # 上一step的可用能<10%且有车在充电站,强制充电动作+1
                if Energy_sources_t_1[n] <= self.IDLE_DISCHARGE_threshold and Evs_id_t_1[n] > 0:
                    charge_actions+= 1
                    forced_charge_actions+= 1
            # 闲置 跟上面的一样处理
            elif action[n] == self.IDLE:
                # 上一step的可用能<10%且有车在充电站,强制充电动作+1
                if Energy_sources_t_1[n] <= self.IDLE_DISCHARGE_threshold and Evs_id_t_1[n] > 0:
                    charge_actions+= 1
                    forced_charge_actions+= 1
        # 风光有剩余且动作要把剩余的充给EV,EV动作为放电功率应尽量小   负载-风光<0且上面得到的动作充电累加>0:
        if houseRW_load_t < 0 and charge_actions > 0:
            # 平均每个充电站理想功率(尽量充电,为正) = -((风光闲置为负-必须放电给负载动作累加*最小给EV充电功率)/充电站充电累加动作)    简单理解: EV充电的理想功率=多出的能量/充电站数
            ch_station_ideal_pwr = -((houseRW_load_t - discharge_actions*self.charging_point_min_power)/charge_actions)
            # 平均充电站理想功率(尽量充电)太大了, 则最大充电动作限制在11
            if ch_station_ideal_pwr > self.charging_point_max_power: ch_station_ideal_pwr = self.charging_point_max_power
            # 平均充电站理想功率(尽量充电)<最小充电限制,则置为最小限制1
            elif ch_station_ideal_pwr < self.charging_point_min_power: ch_station_ideal_pwr = self.charging_point_min_power
        # 风光不够 动作要EV放电来满足用户,EV动作为充电功率应尽量小,EV强制充电按额定功率
        elif houseRW_load_t > 0 and discharge_actions > 0:
            # 平均充电站理想功率(尽量放电,值为负)=-(还需负载正值+如SoC10%以下动作为强制充电相当于负载*单个充电站额定充电功率3.7+EV动作为不强制的正常充电*充电最小功率1)/放电累加动作
            ch_station_ideal_pwr = -((houseRW_load_t + forced_charge_actions*self.charging_point_rated_power + (charge_actions-forced_charge_actions)*self.charging_point_min_power)/discharge_actions)
            # 放电功率太大了,限制在最大功率
            if ch_station_ideal_pwr < -self.charging_point_max_power: ch_station_ideal_pwr = -self.charging_point_max_power
            # 放电功率太小了,限制在最小功率
            elif ch_station_ideal_pwr > -self.charging_point_min_power: ch_station_ideal_pwr = -self.charging_point_min_power
        # houseRW_load_t=0的情况,充电站理想充电功率默认为0
        #__END__ SECTION 0

        # 1检查当前连接的电动车是否离开了充电站,离开了就去掉充电事件  
        #SECTION 1. Check if current connected EVs left the charging station
        # []没有事件就直接跳过了
        for charging_event in self.current_charging_events:
            leaving_time_i = charging_event.leaving_time
            # 车已走
            if time_step >= leaving_time_i:
                # 如果车辆离开，将相应的站点ID设置为零  
                #If vehicle left, set correspondant station ID to zero 
                n = charging_event.station_n
                energy_at_leaving_i, total_ev_power_t = self.apply_action_on_energy_source(step, Energy_sources_t_1[n], action[n], total_ev_power_t, ch_station_ideal_pwr)
                # EVs_energy_at_leaving 就是 Energy_sources_t.append
                self.EVs_energy_at_leaving.append(energy_at_leaving_i)
                new_ev_departures += 1
                Evs_id_t[n] = int(0)
                charging_events_to_remove.append(charging_event)
            else:
                # 如果车辆仍在连接，对应的站点ID=EV的ID.
                #If Vehicle still connected, correspondant station ID = EV's ID
                Evs_id_t[charging_event.station_n] = charging_event.id
        for charging_event in charging_events_to_remove:
            self.current_charging_events.remove(charging_event)
            #__END__ SECTION 1

        #第2节。对每个站段应用行动  
        #SECTION 2. Apply action to each station section 
        for n in range(self.charging_stations_n):
            #(还没看apply_action_on_energy_source) 1. 检查目前的Evs ID，并评估站内可用的新能源n     
            #1. Check Evs id present and evaluate new Energy available at station n
            # 得到剩余的可用能源Energy_sources_t,EV总的充放功率total_ev_power_t
            if Evs_id_t[n] > 0:
                Energy_sources_t[n], total_ev_power_t = self.apply_action_on_energy_source(step, Energy_sources_t_1[n], action[n], total_ev_power_t, ch_station_ideal_pwr)

            elif Evs_id_t[n] == 0:
                #如果在n站没有连接汽车，可用能量=0  
                #If no car is connected at station n, available energy = 0
                if Energy_sources_t[n] != 0:
                    raise ValueError("Available_energy_sources table not matching EVs id: state={} where there is an empty station with a certain energy.".format(Energy_sources_t))
            
            #如果执行了无效的行动，则进行检查，并将其存储在一个表中。 
            #Cheching if invalid actions performed, storing them in a table
            # 动作编码值,用来判断在11个判断4个充电站是否可以充放闲置的列表invalid_actions_t中是否正确.如action_code=4*2+3=11,则0、1、2只能闲置,充电站3只能放,invalid_actions_t=[True, True, True, False, False, False, False, False, False, False, False, True]
            action_code = (self.charging_stations_n*action[n])+n
            # 分别检测第n个充电站的动作是否有效,如n=3,a=2 则2*4+3=11 即action_truth_array的第3个也是最后一个充电站的值=invalid_actions_t的第11个值
            action_truth_array[n] = self.invalid_actions_t[action_code]
            #对无效行为的惩处 
            #Punishment for invalid actions
            #if action_truth_array[n] == False:
            #    reward += -50

            if Energy_sources_t[n] < 0 or Energy_sources_t[n] > 100:
                #检查充电站能量来源是否超出范围.抛出:"可用能源"表超出范围：状态={}不属于状态空间的一部分" 
                #Check if energy sources are out of range
                raise ValueError("Available_energy_sources table out of ranges: state={} which is not part of the state space".format(Energy_sources_t))
            #__END__ SECTION 2

        #检查空闲位置 
        #Checking free spots
        # 充电站可用车位列表[1],...,[0,1,2,3],有可用的就添加其序号. 只有Evs_id_t[n]为0才算可用
        ch_stations_available = []
        for n in range(self.charging_stations_n):
            if Evs_id_t[n] == 0: ch_stations_available.append(n)
        
        charging_events_to_remove = []
        # 3.检查新车是否到达充电站（更新当前电动汽车ID阵列、模拟充电事件列表和当前正在充电的事件列表）。 
        #SECTION 3. Check if new vehicles arrive at charging stations
        for charging_event in self.charging_events:
            arrival_time_i = charging_event.arrival_time
            #将到达时间固定在第0步，改变到第1步  
            #Fixing arrival time at step 0, shifted to step 1
            if step == 1:
                # 若是第一步,且2022.1.1 0:00到达,到达时间设成2022.1.1 0:15
                if arrival_time_i == self.elvis_time_serie[0]:
                    arrival_time_i = self.elvis_time_serie[1]
            # 到达时间在当前时间序列之前,
            if arrival_time_i <= time_step:
                # 如果有可用充电位???
                if len(ch_stations_available)>0:
                    #如果有空闲的电站，从列表中随机弹出空闲的ch_station，并将车辆ID分配给电站。
                    #If free stations available, pop out free ch_station from list and assign vehicle ID to station
                    # pop不生成新list, 对index进行操作，使用随机数randrange（）保证不受参数影响
                    n = ch_stations_available.pop(randrange(len(ch_stations_available)))
                    #if type(charging_event.id) != int:
                    # 即id: 'Charging event: 1306',只取16位后面的数字
                    charging_event.id = int(charging_event.id[16:])
                    Evs_id_t[n] = charging_event.id
                    # 即 battery:...   brand:'Tesla' model: 'Model S' 转化为字典类型
                    vehicle_i = charging_event.vehicle_type.to_dict()
                    soc_i = charging_event.soc
                    # battery:容量、最大小充电功率、效率、开始功率衰减、最大衰减水平
                    battery_i = vehicle_i['battery']
                    # 现在还没实施
                    #efficiency_i  = battery_i['efficiency'] #Not implemented right now
                    #  100kWh
                    capacity_i  = battery_i['capacity'] #kWh
                    #capacity_i  = 100 #kWh, considering only Tesla Model S
                    energy_i = soc_i * capacity_i #kWh
                    # 低于仿真中的最小容量（0.1千瓦时）。
                    if energy_i < 0.1: #Less than min capacity (kWh) in simulation
                        energy_i = 0.1
                    Energy_sources_t[n] = energy_i
                    # 给EV事件赋值在哪充电
                    charging_event.station_n = n
                    # 分别添加到当前正在充电的事件集、要移除的事件集(将来从总的事件中删掉)
                    self.current_charging_events.append(charging_event)
                    charging_events_to_remove.append(charging_event)
                    #break
            elif arrival_time_i > time_step:
                break
        # 从总的事件中删掉
        for charging_event in charging_events_to_remove:
            self.charging_events.remove(charging_event)
        # 可用EV数=充电站数-空闲充电站数
        self.avail_EVs_n[step] = self.charging_stations_n - len(ch_stations_available)
        #__END__ SECTION 3

        # 4更新VPP环境状态和价值 
        #SECTION 4. VPP States and Values updates
        # 当前step的EV需要的总功率(除4就是总能耗),充电为正放电为负
        self.ev_power[step] = total_ev_power_t
        # 住宅去除风光还需负荷功率（还需功率为负载为正）+EV总功率（充电为负载为正） 
        # 一共35041个，如果当前step的值为0可能是EV正好全消耗掉了剩余的风光？？？ 
        self.total_load[step] = houseRW_load_t + total_ev_power_t
        # 还需负载且电价正常,    电价加噪声后可能为负,则消费为0
        if self.total_load[step] > 0 and self.prices_serie[step] > 0:
            # 买电的总消费
            self.overcost[step] = self.total_load[step] * self.prices_serie[step] / 4
        else: self.overcost[step] = 0
        # 买卖电的总消费，负载有正负，电价也有正负
        self.total_cost[step] = self.total_load[step] * self.prices_serie[step] / 4
        self.avail_EVs_id[step] = Evs_id_t
        # EV的能量=SoC*容量
        self.energy_resources[step] = Energy_sources_t
        #评估每一步奖励 
        #Evaluate step reward
        # 括号里的reward=0没用到, 包括EV离开车内剩余能源奖励和负载是否为0(EV正好能满足用户和其他EV)的奖励
        reward = self.eval_reward(reward, step, new_ev_departures)
        #VPP Table UPDATE
        # 存储生成的动作
        self.VPP_actions.append(action)
        # 根据SoC分别确定动作[0,0,0,2]*4+n是否在invalid_actions=[3, 4, 5...10]中的(不可以的行动,在则为False,如0～2充电站没车不能充放,3站有车但SoC高只能放电,所有解释请看mac或定义无效动作那),得到invalid_actions_t=[T, T, T, F, ...F, T]这个列表,最后得到[T,T,T,T]
        self.action_truth_list.append(action_truth_array)
        #self.lstm_states_list.append(self.lstm_state)
        #States UPDATE
        self.state['Available_energy_sources'] = Energy_sources_t
        #self.state['Available_evs_id'] = Evs_id_t #[DELETED]

        ev_power_state = np.zeros(1,dtype=np.float32)
        # 把总的给第0行,只有一个元素
        ev_power_state[0] = total_ev_power_t
        self.state['ev_power'] = ev_power_state
        load_state = np.zeros(1,dtype=np.float32)
        # 把总的给第0行,只有一个元素
        load_state[0] = self.total_load[step]
        self.state['total_load'] = load_state
        #cost_state = np.zeros(1,dtype=np.float32) #[DELETED]
        #cost_state[0] = self.total_cost[step] #[DELETED]
        #self.state['total_cost'] = cost_state #[DELETED]
        #__END__ SECTION 4

        # 检查VPP是否已经完成 
        #FINAL_SECTION: Check if VPP is done
        if self.vpp_length <= 1:
            self.done = True
            # 全置0闲置,用于下一episode?
            self.VPP_actions.append(np.zeros(self.charging_stations_n, dtype=np.int32))
            # 没找到咋用的(除了已注释掉的动作越限负奖励)  action_truth_list、action_truth_array、invalid_actions_t  全置True,用于下一episode?
            self.action_truth_list.append(np.ones(self.charging_stations_n, dtype = bool))
            #self.lstm_states_list.append(None)
            #评估负荷总和（超耗、欠耗）、std和平均数，直到时间步长t，以便进一步奖励 
            #Evaluating load sum (overconsumed, underconsumed), std and average up to timestep t for further rewards
            # 功率
            for load in self.total_load:
                self.sim_total_load += load/4 #kWh
                # 买电
                if load >= 0: self.overconsumed_en += load/4 #kWh
                # 未消纳的风光
                elif load < 0: self.underconsumed_en -= load/4 #kWh
            # total_load均值
            self.sim_av_total_load = np.mean(self.total_load)
            # 标准差
            self.sim_std_total_load = np.std(self.total_load)
            # 即只买电的总消费求和
            self.sim_overcost = np.sum(self.overcost)
            # 买卖电的总消费
            self.sim_total_cost = np.sum(self.total_cost)
            # 1302个EV离开时平均电量
            self.av_EV_energy_left = np.mean(self.EVs_energy_at_leaving)
            self.std_EV_energy_left = np.std(self.EVs_energy_at_leaving)
            charging_events_n = len(self.EVs_energy_at_leaving)
            # 从中一个个remove的,有剩的就错了
            charging_events_left = len(self.charging_events)
            # 检索VPP负载数据框架，以评估自给自足和风光自消纳。 
            VPP_loads = self.VPP_loads #Retrieving the VPP loads Dataframe to evaluate autarky and self-consump.
            # 仅充电   充放功率感觉差不多
            VPP_loads["charging_ev_power"] = self.charging_ev_power
            VPP_loads["discharging_ev_power"] = self.discharging_ev_power

            #可再生资源-自给自足的评价部分 
            #用户负载消耗. 可再生能源未满足用户负载的部分  
            #RENEWABLE-SELF-CONSUMPTION evaluation section
            #Households consump. energy not covered from the Renewables
            # 临时变量=负载-风光>0则都消纳了,<0就有风光剩余未消纳
            VPP_loads["house_self-consump."] = VPP_loads["household_power"] - VPP_loads["RE_power"]
            # 风光未满足用户负载的部分  过滤只剩正值 注意等号后面的值没覆盖仍有正负  
            VPP_loads["RE-uncovered_consump."] = VPP_loads["house_self-consump."].mask(VPP_loads["house_self-consump."].lt(0)).fillna(0) #Filter only positive values
            # 风光未满足用户负载的部分求和/4 kWh
            self.house_uncovered_RE = self.VPP_loads["RE-uncovered_consump."].sum()/4 #kWh
            #家庭直接使用的可再生能源的能源 
            #Energy from the Renewables directly used by the households
            # 风光提供给用户负载的部分=负载-风光未满足负载的部分(正)
            VPP_loads["house_self-consump."] = VPP_loads["household_power"] - VPP_loads["RE-uncovered_consump."]
            # 风光提供给用户负载的部分求和/4
            self.VPP_house_selfc = VPP_loads["house_self-consump."].sum()/4 #kWh
            #Energy from the Renewables exported to the grid
            # 用户未使用的风光
            VPP_loads["house-unused-RE-power"] = VPP_loads["RE_power"] - VPP_loads["house_self-consump."]
            # 临时变量=仅EV充电功率-用户未使用的风光  >0充EV风光还不够都消纳了还需补充的部分, <0就有风光有剩余未消纳部分即还闲置的要上网
            VPP_loads["self_EV-charging"] = VPP_loads["charging_ev_power"] - VPP_loads["house-unused-RE-power"]
            # 加上EV后风光仍未消纳的即给电网的(为正).  过滤只剩负值(还闲置的要上网)后再取相反数 
            VPP_loads["RE-grid-export"] = - VPP_loads["self_EV-charging"].mask(VPP_loads["self_EV-charging"].gt(0)).fillna(0) #Filter only negative values
            # 加上EV后风光仍未消纳的即给电网的(为正)求和/4
            self.RE_grid_export = VPP_loads["RE-grid-export"].sum()/4 #kWh
            #直接储存在EV中的来自可再生能源的能量 
            #Energy from the Renewables directly stored in the EVs batteries
            # >0 充EV风光还不够都消纳了还需补充的部分   过滤只剩正值
            VPP_loads["RE-uncovered_EV-charging"] = VPP_loads["self_EV-charging"].mask(VPP_loads["self_EV-charging"].lt(0)).fillna(0) #Filter only positive values
            # 风光给EV的充电功率=仅EV充电功率-充EV风光还不够都消纳了还需补充的部分
            VPP_loads["self_EV-charging"] = VPP_loads["charging_ev_power"] - VPP_loads["RE-uncovered_EV-charging"]
            # 风光给EV的充电部分求和/4
            self.VPP_RE2battery = VPP_loads["self_EV-charging"].sum()/4 #kWh

            #EV放电功率自消纳评估  
            #用户负载消耗. 电网输入(电动车放电和可再生能源都未满足的能量) 
            #EV-DISCHARGING-Power-SELF-CONSUMPTION evaluation section
            #Households consump. grid import (Energy not covered from the EVs discharging power and Renewables)
            # 临时变量=风光未满足负载的部分(正)-EV仅放电. >0 EV放电仍补充不够负载的部分还需电网的部分. <0 EV放多了能满足负载可上网的部分(负)。
            VPP_loads["battery-self-consump."] = VPP_loads["RE-uncovered_consump."] - (-VPP_loads["discharging_ev_power"]) #THe discharging EV power is a negative serie
            # EV放电仍补充不够负载的部分还需电网的部分. 只剩正值
            VPP_loads["house-grid-import"] = VPP_loads["battery-self-consump."].mask(VPP_loads["battery-self-consump."].lt(0)).fillna(0) #Filter only positive values
            # EV放电给用户的
            self.house_grid_import = VPP_loads["house-grid-import"].sum()/4 #kWh
            #Energy from the EVs discharging power used by the households
            # EV放电给负载的部分=风光未满足负载的部分(正) - EV放电仍补充不够负载的部分还需电网的部分(正)
            VPP_loads["battery-self-consump."] = VPP_loads["RE-uncovered_consump."] - VPP_loads["house-grid-import"]
            # EV放电给电网的(求battery-grid-export,正值)
            self.VPP_battery_selfc = VPP_loads["battery-self-consump."].sum()/4 #kWh
            #Energy from the EVs discharging power, exported to the grid
            # 临时变量EV放电没给负载的部分(给电网或其他EV) = EV放电(取正)-EV放电给负载的部分 . 
            VPP_loads["self_battery-EV-charging"] = (-VPP_loads["discharging_ev_power"]) - VPP_loads["battery-self-consump."] #THe discharging EV power is a negative serie
            # 临时变量EV放电买卖电量=充EV风光还不够都消纳了还需补充的部分(需电网或其他EV)-临时变量EV放电没给负载的部分(给电网或其他EV) >0买电 <0卖电 
            VPP_loads["self_battery-EV-charging"] = VPP_loads["RE-uncovered_EV-charging"] - VPP_loads["self_battery-EV-charging"] #ChargingEVs energy not from renwables - (EVs discharging power not used for the house)
            # EV放电给电网的(正)=-EV放电买卖电量中卖的部分   过滤只剩负值取相反数 
            VPP_loads["battery-grid-export"] = - VPP_loads["self_battery-EV-charging"].mask(VPP_loads["self_battery-EV-charging"].gt(0)).fillna(0) #Filter only negative values
            # 电网给其他EV的 
            self.battery_grid_export = VPP_loads["battery-grid-export"].sum()/4 #kWh
            #Energy from the grid stored in other EVs batteries
            # EV放电电网给其他EV的(正)=EV放电买卖电量中买的部分 只剩正值
            VPP_loads["grid-import_EV-charging"] = VPP_loads["self_battery-EV-charging"].mask(VPP_loads["self_battery-EV-charging"].lt(0)).fillna(0) #Filter only positive values
            self.EVs_grid_import = VPP_loads["grid-import_EV-charging"].sum()/4 #kWh
            #Energy from the EVs discharging power stored in other EVs batteries
            # EV放电给其他EV的 = 充EV风光还不够都消纳了还需补充的部分(需电网或其他EV)-EV放电电网给其他EV的(正)
            VPP_loads["self_battery-EV-charging"] = VPP_loads["RE-uncovered_EV-charging"] - VPP_loads["grid-import_EV-charging"]
            self.VPP_EV2battery = VPP_loads["self_battery-EV-charging"].sum()/4 #kWh
            # 比率评估
            #Rates evaluation
            # 总需求(住宅EV包括买电)
            self.VPP_energy_consumed = self.house_grid_import + self.EVs_grid_import + (self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery)
            # 住宅和EV消纳部分/总需求
            self.VPP_autarky_rate = ((self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery) / self.VPP_energy_consumed) * 100
            # 总能源产出(风光EV包括卖电)
            self.VPP_energy_produced = self.RE_grid_export + self.battery_grid_export + (self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery)
            # 住宅和EV消纳部分/总能源产出
            self.VPP_selfc_rate = ((self.VPP_house_selfc + self.VPP_battery_selfc + self.VPP_RE2battery + self.VPP_EV2battery) / self.VPP_energy_produced) * 100

            #存储修改后的VPP负载数据集
            #Storing the modified VPP loads Dataframe
            self.VPP_loads = VPP_loads
            #最终奖励评估
            #Final reward evaluation
            # 在仿真结束时评估最终的agent奖励,非单步奖励,作为最后一步的奖励
            reward = self.eval_final_reward(reward)
            print("- VPP.Simulation results\n",
                # 1除去风光后的住户和EV负荷需求总数有正负 
                "LOAD_INFO: Sum_Energy=KWh ", round(self.sim_total_load,2),
                # 2有EV后买电网的电 
                f", \nGrid_used_en(grid-import)={round(self.overconsumed_en,2)}kWh",
                # 总需求(住宅EV包括买电)
                f", \nTotal_demand={round(self.VPP_energy_consumed,2)}kWh",
                # 3有EV自给自足率=住宅和EV消纳部分/总需求
                f", \nautarky-rate={round(self.VPP_autarky_rate,1)}",
                # 4有EV多的上电网的闲置能源   total_load中<0的部分
                f", \nRE-to-vehicle_unused_en(grid-export)={round(self.underconsumed_en,2)}kWh",
                # 总能源产出(风光EV包括卖电)
                f", \nTotal_supply={round(self.VPP_energy_produced,2)}kWh",
                # 5有EV自消纳率=住宅EV消纳的风光/总能源产出
                f", \nself-consump.rate={round(self.VPP_selfc_rate,1)}",
                # 6有EV除去风光后买卖电的钱总数
                ", \nTotal_selling_cost=€ ", round(self.sim_total_cost,2),
                # 7只买电的钱只为正
                ", \nGrid_cost=€ ", round(self.sim_overcost,2),
                "\n\n",
                # 1302个EV离开时平均电量
                "EV_INFO: Av.EV_energy_leaving=kWh ", round(self.av_EV_energy_left,2),
                ", \nStd.EV_energy_leaving=kWh ", round(self.std_EV_energy_left,2),
                # 来充电的已离开数量
                ", \nEV_departures = ", charging_events_n,
                # 最后还没走的   从中一个个remove的,有剩的就错了
                ", \nEV_queue_left = ", charging_events_left,"\n")

        else:
            self.done = False

        # 奖励记录上一step的执行情况评估的奖励
        self.reward_hist[step-1] = reward
        #建立最后的表格
        #Building final tables
        if self.done == True:
            self.optimized_VPP_data = pd.DataFrame({'time':self.elvis_time_serie, "rewards":self.reward_hist, "ev_power":self.ev_power, "total_load":self.total_load, "total_cost":self.total_cost, "overcost":self.overcost})
            self.optimized_VPP_data = self.optimized_VPP_data.set_index("time")
            
            self.action_truth_table = np.stack(self.action_truth_list)
            self.Evs_id_table = np.stack(self.avail_EVs_id)
            self.VPP_energies = np.stack(self.energy_resources)
            self.VPP_table = pd.DataFrame(self.VPP_energies)
            self.VPP_table["time"] = self.elvis_time_serie
            self.VPP_table = self.VPP_table.set_index("time")
            self.VPP_table["EVs_id"] = self.avail_EVs_id
            self.VPP_table["actions"] = self.VPP_actions
            self.VPP_table["mask_truth"] = self.action_truth_list
            self.VPP_table["ev_charged_pwr"] = self.charging_ev_power
            self.VPP_table["ev_discharged_pwr"] = self.discharging_ev_power
            self.VPP_table["load"] = self.total_load
            self.VPP_table["load_reward"] = self.load_reward_hist
            self.VPP_table["EV_reward"] = self.EVs_reward_hist
            self.VPP_table["rewards"] = self.reward_hist
            #self.VPP_table["states"] = self.lstm_states_list
            self.cumulative_reward = np.sum(self.reward_hist)
            self.load_t_reward = np.sum(self.load_reward_hist)
            self.EVs_energy_reward = np.sum(self.EVs_reward_hist)
            self.quick_results = np.array([str(self.EVs_n)+"_EVs", self.underconsumed_en, self.overconsumed_en, self.sim_overcost, self.av_EV_energy_left, self.cumulative_reward])
            # Cumulative_reward= 96144.31 - Step_rewards (load_t= 127306.29, EVs_energy_t= -13233.97)
            print(f"SCORE:  Cumulative_reward= {round(self.cumulative_reward,2)} - Step_rewards (load_t= {round(self.load_t_reward,2)}, EVs_energy_t= {round(self.EVs_energy_reward,2)})\n",
                    f"- Final_rewards (Av.EVs_energy= {round(self.AV_EVs_energy_reward,2)}, Grid_used_en= {round(self.overconsume_reward,2)}, RE-to-vehicle_unused_en= {round(self.underconsume_reward,2)}, Grid_cost= {round(self.overcost_reward,2)})\n")
            #__END__ FINAL SECTION
        #设置信息的占位符 
        #set placeholder for info
        info = {}
        #返回执行步后的信息
        #return step information
        return self.state, reward, self.done, info

    def render(self, mode = 'human'):
        """
        Rendering function not implemented.
        """
        #implement visualization
        pass

    def reset(self):
        """
        重置环境功能，为新的仿真做准备
        1. 为电动汽车充电事件创建新的ELVIS仿真
        2. 重置VPP仿真数据集系列，在切除的原始数据集实例上应用噪声（不覆盖）。
        3. 重置VPP仿真表和列表为零或空，以待填充
        Reset Environment function to be ready for new simulation. Divided in 3 main sections:
        - 1. Create new ELVIS simulation for EVs charging events
        - 2. Reset VPP simulation dataset series applying noise on the excrated original dataset instances (not overwriting)
        - 3. Reset VPP simulation tables and lists to zero or empty to be filled
        """
        #SECTION 1. Create new ELVIS simulation
        # 输入EV参数，调用Elvis,返回的EV抵达的分布
        elvis_config_file = self.elvis_config_file
        # 车的抵达分布和开始结束调度周期
        elvis_realisation = elvis_config_file.create_realisation(self.start, self.end, self.res)
        # 1304个事件
        self.charging_events = elvis_realisation.charging_events
        # 初始化中定义的空
        if self.current_charging_events != []:
            current_charging_events = self.current_charging_events
            for i in range(len(current_charging_events)):
                # 相对增量  之前的到达时间2022离开2023，修改后离开时间变成2022
                # 如：d = datetime.date.today()  2022-06-22
                # date1 = d - relativedelta(months=1)  2022-05-22
                # date2 = d - relativedelta(years=1)  2021-06-22
                # date3 = d - relativedelta(days=1)  2022-06-21
                current_charging_events[i].leaving_time = current_charging_events[i].leaving_time - relativedelta(years=1)
            self.current_charging_events = current_charging_events
        # 1304
        self.simul_charging_events_n = len(self.charging_events)
        #Evaluate av.EV energy left with Elvis
        Elvis_av_EV_energy_left, n_av = [0, 0]
        self.EVs_energy_at_arrival = []
        for charging_event in self.charging_events:
            # EV数量
            n_av += 1
            vehicle_i = charging_event.vehicle_type.to_dict()
            soc_i = charging_event.soc
            battery_i = vehicle_i['battery']
            # 现在没有实现
            #efficiency_i  = battery_i['efficiency'] #Not implemented right now
            capacity_i  = battery_i['capacity'] #kWh
            #capacity_i  = 100 #kWh, considering only Tesla Model S
            energy_i = soc_i * capacity_i #kWh
            self.EVs_energy_at_arrival.append(energy_i)
            # 单位秒，不到24h
            charging_time = charging_event.leaving_time - charging_event.arrival_time
            # 300多kwh，容量超了
            final_energy = energy_i + ((charging_time.total_seconds()/3600) * self.charging_point_max_power)
            # 超了改成100kwh最大
            if final_energy > capacity_i: final_energy = capacity_i #kWh
            # EV离开时，EV内的平均剩余能量
            Elvis_av_EV_energy_left = (final_energy + (n_av-1)*Elvis_av_EV_energy_left)/n_av 
        self.Elvis_av_EV_energy_left = Elvis_av_EV_energy_left
        # 对原有数据进行修改添加噪声, 填充EV等数据，下面都是，然后再替换原数据
        VPP_data = self.VPP_data
        #self.prices_serie = VPP_data["EUR/kWh"].values #EUR/kWh #[DELETED]
        #__END__ SECTION 1
        
        #2.向原始数据添加噪声来reset。
        #SECTION 2. Reset VPP simulation data applying noise on the original dataset
        #Data remaining constant: VPP_data["household_power"], VPP_data["solar_power"], VPP_data["wind_power"], VPP_data["EUR/kWh"]:
        mu, sigma = 0, (self.max_energy_price/100)*7 #Mean, standard deviation (self.max_energy_price= 0.13 €/kWh --> 7% = 0.0091)
        # 噪声序列
        price_noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        mu, sigma = 0, (self.houseRWload_max/100)*4 #Mean, standard deviation (self.houseRWload_max= 10kW --> 2% = 0.4 kW)
        load_noise = np.random.normal(mu, sigma, self.tot_simulation_len) # creating a noise with the same dimension of the dataset length
        self.VPP_loads["solar_power"] = VPP_data["solar_power"] - load_noise/3
        self.VPP_loads["wind_power"] = VPP_data["wind_power"] - load_noise/3
        self.VPP_loads["RE_power"] = self.VPP_loads["solar_power"] + self.VPP_loads["wind_power"]
        self.VPP_loads["household_power"] = VPP_data["household_power"] + load_noise/3

        VPP_data["House&RW_load"] = (VPP_data["household_power"] - VPP_data["solar_power"] - VPP_data["wind_power"]) + load_noise
        #Updating series values from noisy table
        self.prices_serie = list(VPP_data["EUR/kWh"].values + price_noise) #EUR/kWh
        self.houseRW_load = VPP_data["House&RW_load"].values  

        with open(os.devnull, "w") as f, contextlib.redirect_stdout(f):
            result = simulate(elvis_realisation)
        VPP_data["ev_power"] = result.aggregate_load_profile(num_time_steps(elvis_realisation.start_date, elvis_realisation.end_date, elvis_realisation.resolution))
        #VPP_data["ev_power"].plot()
        # kwh，15分钟的，注意！！！total_load变成了住宅还需+EV
        VPP_data["total_load"] = VPP_data["House&RW_load"] + VPP_data["ev_power"]
        VPP_data["total_cost"] = VPP_data["total_load"] * VPP_data["EUR/kWh"]/4
        load_array = np.array(VPP_data["total_load"].values)
        cost_array = np.array(VPP_data["total_cost"].values)
        # 只买电的总消费= total_cost > 0求和
        VPP_data["overcost"] = VPP_data["total_cost"]
        VPP_data["overcost"].mask( VPP_data["overcost"] < 0, 0 , inplace=True)
        #Elvis RE2house
        # 负载-风光，过滤负值只剩>=0的。>0则所有风光都给了负荷还不够，<0说明给负载的风光=负载，
        VPP_data["Elvis_RE2house"] = VPP_data["House&RW_load"].mask(VPP_data["House&RW_load"].lt(0)).fillna(0) #Filter only positive values
        # 都为正。House&RW_load>0则负载-负载+风光=给负载的风光。House&RW_load<0则置0即负载-0=负载=给负载的风光
        VPP_data["Elvis_RE2house"] = self.VPP_loads["household_power"] - VPP_data["Elvis_RE2house"]
        self.Elvis_RE2house_en = VPP_data["Elvis_RE2house"].sum()/4 #kWh
        #Elvis Grid2EV
        # 中间变量=只保留(负载-风光)的负值且加负号，结果均为正，闲置的风光。
        VPP_data["Elvis_RE2EV"] = - VPP_data["House&RW_load"].mask(VPP_data["House&RW_load"].gt(0)).fillna(0) #Filter only negative values
        # 中间变量=EV充电功率-闲置的风光，>0的部分为还需电网补，<0为EV充完还剩余的风光
        VPP_data["Elvis_RE2EV"] = VPP_data["ev_power"] - VPP_data["Elvis_RE2EV"]
        # 负的设0，正的保留。
        VPP_data["Elvis_Grid2EV"] = VPP_data["Elvis_RE2EV"].mask(VPP_data["Elvis_RE2EV"].lt(0)).fillna(0) #Filter only positive values
        self.Elvis_Grid2EV_en = VPP_data["Elvis_Grid2EV"].sum()/4 #kWh
        #Elvis RE2EV
        # EV充电功率-电网给EV的，>0
        VPP_data["Elvis_RE2EV"] = VPP_data["ev_power"] - VPP_data["Elvis_Grid2EV"]
        self.Elvis_RE2EV_en = VPP_data["Elvis_RE2EV"].sum()/4 #kWh

        self.av_Elvis_total_load = np.mean(load_array) #kW
        self.std_Elvis_total_load = np.std(load_array) #kW
        self.sum_Elvis_total_load = load_array.sum()/4 #kWh
        # 买电
        self.Elvis_overconsume = load_array[load_array>0].sum()/4 #kWh
        self.Elvis_underconsume = -load_array[load_array<0].sum()/4 #kWh
        # 买卖电都有
        self.Elvis_total_cost = cost_array.sum() #€
        self.Elvis_overcost = cost_array[cost_array > 0].sum()
        # Elvis的自给自足和自消纳评价
        #Elvis self-consumption and autarky eval
        # total_load<0的取正即未消费的风光+风光给住宅+风光给EV    注意！！！total_load变成了住宅还需负荷+EV。
        self.Elvis_en_produced = self.Elvis_underconsume + (self.Elvis_RE2house_en + self.Elvis_RE2EV_en)
        self.Elvis_selfc_rate = ((self.Elvis_RE2house_en + self.Elvis_RE2EV_en) / self.Elvis_en_produced)*100
        # 总能耗=买电+风光给住户的+风光给EV的
        self.Elvis_en_consumed = self.Elvis_overconsume + (self.Elvis_RE2house_en + self.Elvis_RE2EV_en)
        self.Elvis_autarky_rate = ((self.Elvis_RE2house_en + self.Elvis_RE2EV_en) / self.Elvis_en_consumed)*100
        #Reset environment printout:
        # 加上EV后的仿真
        print("- ELVIS.Simulation (Av.EV_SOC= ", self.EVs_mean_soc, "%):\n",
            # 加上EV后总负载
            "Sum_Energy=kWh ", round(self.sum_Elvis_total_load,2),
            # 买电
            f", \nGrid_used_en(grid-import)={round(self.Elvis_overconsume,2)}kWh",
            # 住户+EV总能耗
            f", \nTotal_demand={round(self.Elvis_en_consumed,2)}kWh",
            # 自给自足率
            f", \nautarky-rate={round(self.Elvis_autarky_rate,1)}",
            # 闲置的风光
            f", \nRE-to-vehicle_unused_en(grid-export)={round(self.Elvis_underconsume,2)}kWh",
            # 风光发的总量
            f", \nTotal_supply={round(self.Elvis_en_produced,2)}kWh",
            # 本地消纳率
            f", \nself-consump.rate={round(self.Elvis_selfc_rate,1)}",
            # 买电消费
            ", \nGrid_cost=€ ", round(self.Elvis_overcost,2),
            # 买卖电消费
            ", \nTotal_selling_cost=€ ", round(self.Elvis_total_cost,2),
            # EV充完后平均SoC电量
            ", \nAv.EV_en_left=kWh ", round(Elvis_av_EV_energy_left,2),
            ", \nCharging_events= ", self.simul_charging_events_n,
            "\n- VPP_goal_upper_limit: Grid_used_en=kWh 0, RE-to-vehicle_unused_en=kWh 0, Grid_cost=€ 0",
            # 预计能充EV电量(初始化时算的)
            ", Av.EV_en_left=kWh ",round(self.exp_ev_en_left,2),"\n")
        #__END__ SECTION 2
        
        #3.重置VPP仿真表和待填列表 
        #SECTION 3. Reset VPP simulation tables and lists to be filled
        #Setting reward functions
        self.set_reward_func()
        #重置VPP对话的长度
        #Reset VPP session length
        self.vpp_length = self.tot_simulation_len
        self.energy_resources, self.avail_EVs_id, self.avail_EVs_n, self.ev_power, self.charging_ev_power, self.discharging_ev_power , self.total_cost, self.overcost, self.total_load, self.reward_hist, self.EVs_reward_hist, self.load_reward_hist = ([],[],[],[],[],[],[],[],[],[],[],[])
        #建立EV序列(可用能,ID)   
        #build EV series (Avail_en. and IDs)
        for i in range(len(self.elvis_time_serie)):
            # [0,0,0,0]*35041   4个充电站
            self.energy_resources.append(np.zeros(self.charging_stations_n, dtype=np.float32))
            self.avail_EVs_id.append(np.zeros(self.charging_stations_n, dtype=np.int32))
            # 0  可用EV数量
            self.avail_EVs_n.append(0)
            self.ev_power.append(0.0)
            self.charging_ev_power.append(0.0)
            self.discharging_ev_power.append(0.0)
            self.total_cost.append(0.0)
            self.overcost.append(0.0)
            self.total_load.append(0.0)
            self.reward_hist.append(0)
            self.EVs_reward_hist.append(0)
            # 负载每步的奖励
            self.load_reward_hist.append(0)
        
        # 第0个数据替换一下
        self.total_load[0] = self.houseRW_load[0]
        # 同上
        self.total_cost[0] = self.total_load[0] * self.prices_serie[0]/4
        # 第0列还是0, 即[0,0,0,0]*35041
        self.energy_resources[0] = self.Init_space["Available_energy_sources"]
        #self.avail_EVs_id[0] = self.Init_space['Available_evs_id'] #[DELETED]
        # 将新场景替换原数据以便训练
        self.VPP_data = VPP_data
        # 由RL agent控制的充电行动的VPP仿真结果
        self.optimized_VPP_data = pd.DataFrame({'time':self.elvis_time_serie, "rewards":self.reward_hist, "ev_power":self.ev_power, "total_load":self.total_load, "total_cost":self.total_cost})
        self.optimized_VPP_data = self.optimized_VPP_data.set_index("time")
        #self.lstm_states_list = []
        self.VPP_actions, self.action_truth_list, self.EVs_energy_at_leaving= ([],[],[]) 
        self.av_EV_energy_left, self.std_EV_energy_left, self.sim_total_load, self.sim_av_total_load, self.sim_std_total_load, self.overconsumed_en, self.underconsumed_en, self.sim_total_cost, self.sim_overcost = [0, 0, 0, 0, 0, 0, 0, 0, 0]
        self.cumulative_reward, self.load_t_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward, self.EVs_energy_reward, self.AV_EVs_energy_reward = [0, 0, 0, 0, 0, 0, 0]
        #For plotting battery levels
        #self.VPP_energies = self.Init_space["Available_energy_sources"] #[DELETED]
        # 重置无效动作列表为 11个T
        self.invalid_actions_t = np.ones(len(self.actions_set), dtype=bool)
        self.VPP_table = []
        self.quick_results = []
        #Set starting cond.
        self.state = self.Init_space
        #reset vpp session time
        self.vpp_length = self.tot_simulation_len
        self.done = False
        print("Simulating VPP....")
        print("\n")
        #__END__ SECTION 3
        return self.state
    
    def save_VPP_table(self, save_path='data/environment_optimized_output/VPP_table.csv'):
        """
        保存VPP优化的仿真数据。
        Method to save the VPP optimized simulation data.
        """
        self.VPP_table.to_csv(save_path)
        return self.VPP_table
    
    def plot_ELVIS_data(self):
        """
        ELVIS仿真输入参数。EV分布，平均停车时间、方差，车型号，风、光、负荷、单个充电站充电、住宅除去风光剩余(每多一户+1kW)功率最大值
        Method to plot and visualize the ELVIS simulation input data for the EVs infrastructure.
        """
        # 车到达的周概率分布，基本每天一循环，有图片Elvis_config可查
        #Weekly arrival distribution simulation
        # D:\Anaconda3\envs\RL3\Lib\site-packages\elvis\config.py
        weekly_distribution = self.elvis_config_file.arrival_distribution
        time_frame = self.elvis_time_serie[0:len(weekly_distribution)*4:4]

        EV_battery_capacities,models = ([], [])
        for EV_type in self.EV_types:
            EV_battery_capacities.append(EV_type["battery"]["capacity"])
            #brand.append()
            models.append(str(EV_type['brand'])+str(EV_type['model']))
        
        Elvis_data_fig = make_subplots(subplot_titles=('EVs arrival distribution (weekly)','Simulation parameters', 'EV models', 'Rated powers'),
                            rows=2, cols=2,
                            specs=[[{"secondary_y": False},{"type": "table"}],
                                    [{"secondary_y": False},{"secondary_y": False}]])
        
        Elvis_data_fig.add_trace(
            go.Scatter(x=time_frame, y=weekly_distribution, name="EVs_arrival distribution"),
            row=1, col=1, secondary_y=False)
        
        table_data = [['EV_arrivals(W)','mean_park(h)','mean_park+std','mean_park-std'],[self.EVs_n, self.mean_park, self.mean_park+self.std_deviation_park, self.mean_park-self.std_deviation_park]]
        Elvis_data_fig.add_trace(go.Table(
                                    columnorder = [1,2],
                                    columnwidth = [80,400],
                                    header = dict(
                                        values = [['Parameters'],
                                                    ['Values']],
                                        fill_color='#04cc98',
                                        align=['left','center'],
                                    ),
                                    cells=dict(
                                        values=table_data,
                                        fill=dict(color=['royalblue', 'white']),
                                        align=['left', 'center'],
                                        #height=30
                                    )), row=1, col=2)
                                        
        Elvis_data_fig.add_trace(go.Bar(x=[models[0],'arrival Av.soc','Av.soc-std','Av.soc+std'], y=[EV_battery_capacities[0], self.EVs_mean_soc, (self.EVs_mean_soc-self.EVs_std_deviation_soc), (self.EVs_mean_soc+self.EVs_std_deviation_soc)], marker_color = ['#d62728','#bcbd22','#7f7f7f','#7f7f7f']),
                            row=2, col=1)
        
        rated_powers_x = ['solar max', 'wind max', 'EVs load max', 'ch.point max', 'houseRWload max']
        rated_powers_y = [self.solar_power, self.wind_power, self.EV_load_max, self.charging_point_max_power, self.houseRWload_max]
        marker_color = ['#95bf00', '#1ac6ff', '#ee5940', '#7f7f7f', 'orange']
        
        Elvis_data_fig.add_trace(go.Bar(x=rated_powers_x, y=rated_powers_y, marker_color=marker_color),
                            row=2, col=2)
        
        Elvis_data_fig['layout']['yaxis1'].update(title='Probability')
        Elvis_data_fig['layout']['yaxis2'].update(title='Battery capacity (kWh)')
        Elvis_data_fig['layout']['yaxis3'].update(title='kW')
        #Elvis_data_fig['layout']['legend'].update(title=f'Cumulat.Reward= {round(self.cumulative_reward,2)}')
        Elvis_data_fig.update_layout(title_text='ELVIS simulation input data', width=1500,height=550, showlegend = False)
        return Elvis_data_fig
    
    def plot_VPP_input_data(self):
        """
        绘制和可视化VPP环境输入数据集的方法：住宅负荷、风、光、住宅还需负荷、电价随时间的曲线1年
        Method to plot and visualize the VPP environment input dataset.
        """
        #Optimized VPP simulation graphs
        VPP_data_fig = make_subplots(
                            subplot_titles=('Households and Renewables power over time','Households+RW sources Load over time','Energy cost over time'),
                            rows=3, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": False}],
                                    [{"secondary_y": False}],
                                    [{"secondary_y": False}]])

        # Top graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="住宅负荷household_power",line={'color':'#5c5cd6'}, stackgroup='consumed'),
            row=1, col=1, secondary_y=False)
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="光solar_power",line={'color':'#95bf00'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="风wind_power",line={'color':'#1ac6ff'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        
        # Center graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["House&RW_load"], name="住宅-风光House&RW_load",line={'color': 'orange'}, stackgroup='summed'),
            row=2, col=1, secondary_y=False)

        #Down graph
        VPP_data_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["EUR/kWh"], name="EUR/kWh",line={'color':'rgb(210, 80, 75)'}, stackgroup='cost'),
            row=3, col=1, secondary_y=False)

        VPP_data_fig['layout']['yaxis1'].update(title='kW')
        VPP_data_fig['layout']['yaxis2'].update(title='kW')
        VPP_data_fig['layout']['yaxis3'].update(title='€/kWh')
        VPP_data_fig['layout']['legend'].update(title='Time series')
        VPP_data_fig.update_layout(title_text='VPP simulation input data', width=1500,height=700, xaxis3_rangeslider_visible=True, xaxis3_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=1, col=1)
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=2, col=1)
        VPP_data_fig.update_xaxes(range=["2022-06-01 00:00:00", "2022-06-11 00:00:00"], row=3, col=1)        
        #VPP_data_fig.show()
        return VPP_data_fig
    
    def plot_reward_functions(self):
        """
        绘制并可视化RL agent的奖励函数
        Method to plot and visualize the RL agent reward functions.
        """
        #Step rewards
        battery_x = np.linspace(0, 100, 200)
        battery_y = np.interp(battery_x, self.battery_percentage, self.EVs_energy_reward_range)

        load_x = np.linspace(self.load_range[0], self.load_range[-1], 10000)
        load_y = np.interp(load_x, self.load_range, self.load_reward_range)

        #Final rewards
        final_battery_y = np.interp(battery_x, self.av_energy_left_range, self.av_energy_reward_range)

        overconsume_x = np.linspace(self.overconsume_range[0], self.overconsume_range[-1], 200)
        overconsume_y = np.interp(overconsume_x, self.overconsume_range, self.overconsume_reward_range)

        underconsume_x = np.linspace(self.underconsume_range[0], self.underconsume_range[-1], 200)
        underconsume_y = np.interp(underconsume_x, self.underconsume_range, self.underconsume_reward_range)

        cost_x = np.linspace(self.overcost_range[0], self.overcost_range[-1], 200)
        cost_y = np.interp(cost_x, self.overcost_range, self.overcost_reward_range)

        rewards_fig = make_subplots(subplot_titles=('Step EVs energy (when leaving) reward f.','Step load reward f.', 'Final Grid energy used reward f.', 'Final Av.EVs-departure energy reward f.', 'Final Overcost reward f.', 'Final RE-to-vehicle unused energy reward f.'),
                            rows=2, cols=3,
                            specs=[[{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}],
                                    [{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])
        
        rewards_fig.add_trace(go.Scatter(x=battery_x, y=battery_y, name="step_ev_energy", stackgroup='1'),
                            row=1, col=1, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=load_x, y=load_y, name="step_load", stackgroup='1'),
                            row=1, col=2, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=overconsume_x, y=overconsume_y, name="final_Grid_used_en", stackgroup='1'),
                            row=1, col=3, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=battery_x, y=final_battery_y, name="final_Av.ev_energy", stackgroup='1'),
                            row=2, col=1, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=cost_x, y=cost_y, name="final_Grid-cost", stackgroup='1'),
                            row=2, col=2, secondary_y=False)
        rewards_fig.add_trace(go.Scatter(x=underconsume_x, y=underconsume_y, name="final_RE-to-vehicle_unused_en", stackgroup='1'),
                            row=2, col=3, secondary_y=False)
        
        
        rewards_fig['layout']['xaxis1'].update(title='Battery% (kWh)')
        rewards_fig['layout']['xaxis2'].update(title='kW')
        rewards_fig['layout']['xaxis3'].update(title='kWh')
        rewards_fig['layout']['xaxis4'].update(title='Battery% (kWh)')
        rewards_fig['layout']['xaxis5'].update(title='€')
        rewards_fig['layout']['xaxis6'].update(title='kWh')
        rewards_fig['layout']['yaxis1'].update(title='Step reward')
        rewards_fig['layout']['yaxis2'].update(title='Step reward')
        rewards_fig['layout']['yaxis3'].update(title='Final reward')
        rewards_fig['layout']['yaxis4'].update(title='Final reward')
        rewards_fig['layout']['yaxis5'].update(title='Final reward')
        rewards_fig['layout']['yaxis6'].update(title='Final reward')
        rewards_fig.update_layout(title_text='Reward functions', width=1500,height=700, showlegend = False)
        #rewards_fig.show()
        return rewards_fig
        
    def plot_Dataset_autarky(self):
        """
        在普通数据集中绘制和显示自给自足和自消纳  
        Method to plot and visualize the autarky and self-consumption
        in the plain dataset.

        数据集 自给自足和自我消费。能源计量单位：千瓦时 供应能源：50260.3
        数据集的自我消费率。47.9%，总供应量-en: 50260.3千瓦时

         需求-能量：29045.6千瓦时。
         自给率：83.0%, 总需求能源Tot.demand-en:29045.6kWh
        """
        # 住宅总负荷（不考虑风光） 29045
        en_demand = self.household_consume
        # 总风光  50260
        en_supply = self.RW_energy
        
        # 自消纳率（风光消纳率）=住宅消纳的风光/风光总产出
        selfc_rate = (self.self_consumption / en_supply) * 100
        # 自给自足率=自消纳/住宅总负荷（不考虑风光）, 自消纳=住宅总功耗不考虑风光-从电网买的电
        autarky_rate = (self.self_consumption / en_demand) * 100
        # 闲置的风光    住宅消纳的风光
        selfc_labels = ["RE2grid-export", "RE2house-self"]
        # 风光够负荷用后多的转化为EV的闲置能源
        selfc_values = [-self.HRW_underenergy, self.self_consumption]
        # 买电  住宅消纳的风光   风光不够住宅用电，又从电网买的>0求和， 自消纳=住宅总功耗不考虑风光-从电网买的电   
        autarky_labels = ["Grid2house-import", "RE2house-self"]
        autarky_values = [self.HRW_overenergy, self.self_consumption]
        # 创建子图：为Pie子图使用'域'类型  
        # Create subplots: use 'domain' type for Pie subplot

        # Tot.supplied-en风光提供的总量=(-闲置)+自消纳)
        # =-HRW_underenergy+self_consumption
        # =-风光够负荷用后多的转化为EV的闲置能源即houseRW_load<0求和+（自消纳=住宅总功耗不考虑风光-除去风光不够从电网买的电>0求和）
        # =-（不够的部分负荷总-不够的部分风光）+负荷总-（够的部分负荷总-够的部分风光）
        # =-不够的部分负荷总+不够的部分风光+负荷总-够的部分负荷总+够的部分风光
        # =不够的部分风光+够的部分风光
        # =风光总产量
        
        # Tot.demand-en总需求=买电+(自消纳=住宅总功耗不考虑风光-从电网买的电）
        # 消纳率 风光总量
        fig = make_subplots(subplot_titles=(f'Dataset self-consumption rate: {round(selfc_rate,1)}%,  Tot.supplied-en: {round((-self.HRW_underenergy)+self.self_consumption,1)} kWh',
                                            f'Autarky rate: {round(autarky_rate,1)}%,  Tot.demand-en:{round(self.HRW_overenergy+self.self_consumption,1)}kWh'),
                            # 自给自足率  总需求
                            rows=1, cols=2, specs=[[{'type':'domain'}, {'type':'domain'}]])
        
        fig.add_trace(go.Pie(labels=selfc_labels, values=selfc_values, name="self-consumption", textinfo='label+value+percent', pull=[0.1, 0]),
                      1, 1)
        fig.add_trace(go.Pie(labels=autarky_labels, values=autarky_values, name="autarky", textinfo='label+value+percent', pull=[0.1, 0]),
                      1, 2)
        
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.2, hoverinfo="label+value+percent")
        
        fig.update_layout(
            # 数据集自给自足和自消纳  总风光  总负荷
            title_text="Data-set Autarky and self-consumption. \nEnergy measuring unit: kWh"+f'\nSupply-energy:{round(en_supply,1)}kWh.     '+ f'\nDemand-energy:{round(en_demand,1)}kWh.',
            # 在甜甜圈馅饼的中心添加注释 
            # Add annotations in the center of the donut pies.
            #annotations=[dict(text=f'Supply-energy:{round(en_supply,1)}kWh', x=0.18, y=0.5, font_size=20, showarrow=False),
            #             dict(text=f'Demand-energy:{round(en_demand,1)}kWh', x=0.82, y=0.5, font_size=20, showarrow=False)],
            width=1500,height=600, # efan 原来width=1500,height=550,
            showlegend = False)

        # .py调用的时候需要，ipynb不需要
        #fig.show()
        return fig
        

    def plot_VPP_autarky(self):
        """
        对比:比较Elvis非控制性充电仿真和有控制性充电行动的VPP仿真中的自给自足和自消纳情况。
        Method to plot and compare the autarky and self-consumption in the Elvis uncontrolled-charging
        simulation and in the VPP simulation with controlled-charging actions.
        """
        Elvis_selfc_labels = ["RE2grid-export", "RE2house-self", "RE2EVs-self"]
        Elvis_selfc_values = [self.Elvis_underconsume, self.Elvis_RE2house_en, self.Elvis_RE2EV_en]
        Elvis_autarky_labels = ["Grid2house-import", "Grid2EV-import", "RE2house-self","RE2EVs-self"]
        Elvis_autarky_values = [(self.Elvis_overconsume-self.Elvis_Grid2EV_en), self.Elvis_Grid2EV_en, self.Elvis_RE2house_en, self.Elvis_RE2EV_en]
        
        VPP_selfc_labels = ["RE2grid-export", "EV2grid-export", "RE2house-self", "RE2EVs-self", "EV2house-self", "EV2EV-transf."]
        VPP_selfc_values = [self.RE_grid_export, self.battery_grid_export, self.VPP_house_selfc, self.VPP_RE2battery, self.VPP_battery_selfc, self.VPP_EV2battery]
        VPP_autarky_labels = ["Grid2house-import", "Grid2EV-import", "RE2house-self", "RE2EVs-self", "EV2house-self", "EV2EV-transf."]
        VPP_autarky_values = [self.house_grid_import, self.EVs_grid_import, self.VPP_house_selfc, self.VPP_RE2battery, self.VPP_battery_selfc, self.VPP_EV2battery]

        # Create subplots
        # fig = make_subplots(subplot_titles=('Elvis simulation', 'Elvis simulation',
        #                       'VPP simulation', 'VPP simulation'),
        #                     rows=2, cols=2,
        #                     specs=[[{'type':'domain'}, {'type':'domain'}],
        #                             [{'type':'domain'}, {'type':'domain'}]])
        fig = make_subplots(subplot_titles=(f'Elvis-Self-consump.rate:{round(self.Elvis_selfc_rate,1)}%,  Tot.supplied-en:{round(self.Elvis_en_produced,1)} kWh', f'Elvis-Autarky-rate:{round(self.Elvis_autarky_rate,1)}%,  Tot.demand-en:{round(self.Elvis_en_consumed,1)}kWh',
                              # 此处4行有简化
                              f'VPP-Self-consumption rate: {round(self.VPP_selfc_rate,1)}%'+'\n'+f'Tot.supplied-en: {round(self.VPP_energy_produced,1)} kWh', f'VPP-Autarky rate: {round(self.VPP_autarky_rate,1)}%\nTot.demand-en:{round(self.VPP_energy_consumed,1)}kWh'),
                            rows=2, cols=2,
                            specs=[[{'type':'domain'}, {'type':'domain'}],
                                    [{'type':'domain'}, {'type':'domain'}]])
        # pull参数设置饼图的各个扇形的突出程度  textinfo参数用于设置在扇形上的具体数值  rotation参数可以对饼图进行旋转，其取值为0-360  showlegend布尔型，True表示展示，False表示隐藏
        fig.add_trace(go.Pie(labels=Elvis_selfc_labels, values=Elvis_selfc_values, name="elvis_self-consumption", textinfo='label+value+percent', pull=[0.1, 0, 0], sort=False, rotation=10),
                      1, 1)
        fig.add_trace(go.Pie(labels=Elvis_autarky_labels, values=Elvis_autarky_values, name="elvis_autarky", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0], sort=False, rotation=140),
                      1, 2)

        fig.add_trace(go.Pie(labels=VPP_selfc_labels, values=VPP_selfc_values, name="VPP_self-consumption", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0, 0, 0], sort=False, rotation=70),
                      2, 1)
        fig.add_trace(go.Pie(labels=VPP_autarky_labels, values=VPP_autarky_values, name="VPP_autarky", textinfo='label+value+percent', pull=[0.1, 0.1, 0, 0, 0, 0], sort=False, rotation=80),
                      2, 2)
        
        # 在饼图里面再抠一个多大的'孔'  
        # Use `hole` to create a donut-like pie chart
        fig.update_traces(hole=.2, hoverinfo="label+value+percent")
        # 在甜甜圈馅饼的中心添加注释 有bug.    showarrow：布尔值，是否添加从标签到数据点的箭头
        # 修改
        fig.update_layout(
            #title_text="Data-set Autarky and self-consumption",
            # Add annotations in the center of the donut pies.
            annotations=[#dict(text='Elvis simulation', x=0.25, y=0.95, font_size=14, showarrow=False),
                         dict(text=f'Self-consumption rate: {round(self.Elvis_selfc_rate,1)}%', x=0.4, y=0.88, font_size=12, showarrow=False),
                         dict(text=f'Tot.supplied-en: {round(self.Elvis_en_produced,1)} kWh', x=0.4, y=0.83, font_size=12, showarrow=False),
                         #dict(text='Elvis simulation', x=0.65, y=0.95, font_size=14, showarrow=False),
                         dict(text= f'Autarky rate: {round(self.Elvis_autarky_rate,1)}%', x=1, y=0.88, font_size=12, showarrow=False),
                         dict(text= f'Tot.demand-en:{round(self.Elvis_en_consumed,1)}kWh', x=1, y=0.83, font_size=12, showarrow=False),
                         #dict(text='VPP simulation', x=0.05, y=0.35, font_size=14, showarrow=False),
                         dict(text=f'Self-consumption rate: {round(self.VPP_selfc_rate,1)}%', x=0.4, y=0.38, font_size=12, showarrow=False),
                         dict(text=f'Tot.supplied-en: {round(self.VPP_energy_produced,1)} kWh', x=0.4, y=0.33, font_size=12, showarrow=False),
                         #dict(text='VPP simulation', x=0.65, y=0.35, font_size=14, showarrow=False),
                         dict(text= f'Autarky rate: {round(self.VPP_autarky_rate,1)}%', x=1, y=0.38, font_size=12, showarrow=False),
                         dict(text= f'Tot.demand-en:{round(self.VPP_energy_consumed,1)}kWh', x=1, y=0.33, font_size=12, showarrow=False)],

            width=1550,height=800,
            showlegend = False)

        #fig.show()
        return fig
    
    def plot_VPP_energies(self):
        """
        在VPP仿真过程中充电站的可用能量水平，并控制充电行动。
        Method to plot and visualize the available energy levels present at the charging stations
        during the VPP simulation with controlled charging actions.
        """
        #绘制n个充电站在一段时间内的可用能量图、EV离开的平均值±标准差
        #Plot energy available in the charging points over time
        self.VPP_energies = pd.DataFrame(self.VPP_energies)
        self.VPP_energies["time"] = self.elvis_time_serie
        self.VPP_energies = self.VPP_energies.set_index("time")
        VPP_energies_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])
        for n in range(self.charging_stations_n):
            station = str(n)
            VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=self.VPP_energies[n], name=f"charging station {station}", stackgroup=f"{station}"),
                                    row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left-self.std_EV_energy_left]*self.tot_simulation_len, line={'color':'lightgrey'},
        name="-Std_EV_energy_left"), row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left+self.std_EV_energy_left]*self.tot_simulation_len, line={'color':'lightgrey'},
        name="+Std_EV_energy_left"), row=1, col=1, secondary_y=False)

        VPP_energies_fig.add_trace(go.Scatter(x=self.elvis_time_serie, y=[self.av_EV_energy_left]*self.tot_simulation_len, line={'color':'#bcbd22'},
        name="Av_EV_energy_left"), row=1, col=1, secondary_y=False)
            
        VPP_energies_fig['layout']['yaxis1'].update(title='kWh')
        VPP_energies_fig.update_layout(title_text='VPP available energies at EV charging points', width=1500,height= 550, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #VPP_energies_fig.show()
        return VPP_energies_fig

    def plot_Elvis_results(self):
        """
        ELVIS不受控制的充电的仿真结果（load, EV power 和overcost即只买电的总消费= total_cost > 0求和 ）
        Method to plot and visualize the ELVIS simulation results (load, EV power and overcost) with uncontrolled charging.
        """
        #Elvis simulation graphs
        Elvis_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": True}]])

        # Top graph
        #Elvis_fig.add_trace(
        #    go.Scatter(x=self.elvis_time_serie, y=[0]*self.tot_simulation_len,line={'color':'#00174f'}, name="zero_load"),
        #    row=1, col=1, secondary_y=False)
        
        """ Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="solar_power",line={'color':'#95bf00'}),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="wind_power",line={'color':'#1ac6ff'}),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="household_power",line={'color':'#5c5cd6'}),
            row=1, col=1, secondary_y=False) """
        
        
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.houseRW_load, line={'color':'orange'}, name="houseRW_load", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            # stackgroup相同的会堆叠，即在之前的曲线上再叠加,而不是以x轴作为起点
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["ev_power"], line={'color':'rgb(77, 218, 193)'}, name="ev_power以住宅为基础绘制", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load"),
            row=1, col=1, secondary_y=False)

        # Down
        Elvis_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["overcost"], line={'color':'rgb(210, 80, 75)'}, name="grid-cost", stackgroup="cost"),
            row=1, col=1, secondary_y=True)

        Elvis_fig['layout']['yaxis1'].update(title='kW')
        Elvis_fig['layout']['yaxis2'].update(title='€')
        Elvis_fig['layout']['legend'].update(title='Time series')
        Elvis_fig.update_layout(title_text='Elvis Load, EVs power, Grid-cost', width=1500,height= 600, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #Elvis_fig.show()
        return Elvis_fig

    def plot_VPP_results(self):
        """
        由RL agent控制的充电行动的VPP仿真结果（35041步结束后记录,由houseRW_load, ev_power and overcost叠加的输入数据集）。
        Method to plot and visualize the VPP simulation results (Input dataset superimposed with load, EV power and overcost)
        with charging actions controlled by the RL agent.
        """
        #Optimized VPP simulation graphs
        VPP_opt_fig = make_subplots(rows=1, cols=1,
                                    #shared_xaxes=True,
                                    specs=[[{"secondary_y": True}]])

        #VPP_opt_fig.add_trace(
        #    go.Scatter(x=self.elvis_time_serie, y=[0]*self.tot_simulation_len,line={'color':'#00174f'}, name="zero_load"),
        #    row=1, col=1, secondary_y=False)
        """ VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["solar_power"], name="solar_power",line={'color':'#95bf00'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_data["wind_power"], name="wind_power",line={'color':'#1ac6ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_data["household_power"], name="household_power",line={'color':'#5c5cd6'}),
            row=1, col=1, secondary_y=False) """
            

        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.houseRW_load, line={'color':'orange'}, name="houseRW_load", stackgroup="power"),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # 在上面曲线上叠加
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["ev_power"], line={'color':'rgb(77, 218, 193)'}, name="ev_power", stackgroup="power"),
            row=1, col=1, secondary_y=False)

        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load"),
            row=1, col=1, secondary_y=False)
        # Down
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["overcost"],line={'color':'rgb(210, 80, 75)'}, name="grid-cost", stackgroup="cost"),
            row=1, col=1, secondary_y=True)

        VPP_opt_fig['layout']['yaxis1'].update(title='kW')
        #VPP_opt_fig['layout']['yaxis2'].update(title='kW')
        VPP_opt_fig['layout']['yaxis2'].update(title='€')
        VPP_opt_fig['layout']['legend'].update(title='Time series')
        VPP_opt_fig.update_layout(title_text='VPP Load, EVs power, Grid-cost', width=1500,height= 600, xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        #VPP_opt_fig.show()
        return VPP_opt_fig
    
    def plot_VPP_supply_demand(self):
        """
        随着时间的推移，VPP的供应/需求能量由RL代理控制的充电行动（充电/放电的EV功率，最终total load叠加的输入数据集）。
        Update plot：使用supply/demand分析自消费/自消纳。
        Method to plot and visualize the VPP supply/demand energy over time (Input dataset superimposed with charging/discharging EV power, resulting total load)
        with charging actions controlled by the RL agent.
        Update plot: supply/demand usage for Self-consumption/autarky analysis
        """
        #Optimized VPP simulation graphs
        VPP_opt_fig = make_subplots(#subplot_titles=(f'Supply/demand sources', f'Supply/demand usage'), shared_xaxes=True,
                                    rows=1, cols=1, 
                                    specs=[[{"secondary_y": False}],
                                            #[{"secondary_y": False}]
                                            ])
        #UP
        #家庭负荷来源 
        #Households consumption power sources
        VPP_opt_fig.add_trace(
            # 风光提供给用户负载的部分
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["house_self-consump."], name="RE2house_self-consump.", stackgroup='positive',line={'color':'#67ff24'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # EV提供给用户负载的部分
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["battery-self-consump."], name="EV2house_self-consump.", stackgroup='positive',line={'color':'#fc24ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # 电网提供给用户负载的部分
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["house-grid-import"], name="Grid2house_import", stackgroup='positive'),
            row=1, col=1, secondary_y=False)
        #EV充电来源 
        #EV charging power sources
        VPP_opt_fig.add_trace(
            # 风光提供给EV充电的部分
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["self_EV-charging"], name="RE2EV_self-consump.", stackgroup='positive',line={'color':'#24f3ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # 其他EV提供给EV充电的部分
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["self_battery-EV-charging"], name="EV2EV_self-consump.", stackgroup='positive',line={'color':'#fff824'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # 电网提供给EV充电的部分功率
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["grid-import_EV-charging"], name="Grid2EV_import", stackgroup='positive'),
            row=1, col=1, secondary_y=False)
        #消费实体 
        #Consumption entities
        VPP_opt_fig.add_trace(
            # 负载功率
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["household_power"], name="CO_household_power",line={'color':'#5c5cd6'}, stackgroup='consumed'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # 仅充电功率
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["charging_ev_power"],line={'color':'#45d3d3'}, name="CO_EV_charging_pwr", stackgroup='consumed'),
            row=1, col=1, secondary_y=False)

        # DOWN
        #使用新能源的功率,注意负号  
        #Renewable produced power usage
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["house_self-consump."], name="RE2house_self-consump.", stackgroup='negative',line={'color':'#67ff24'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["self_EV-charging"], name="RE2EV_self-consump.", stackgroup='negative',line={'color':'#24f3ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["RE-grid-export"], name="RE2grid_export", stackgroup='negative'),
            row=1, col=1, secondary_y=False)
        #使用EV放电的功率  
        #EV discharged power usage
        VPP_opt_fig.add_trace(
            # EV给住宅
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["battery-self-consump."], name="EV2house_self-consump.", stackgroup='negative',line={'color':'#fc24ff'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # EV给EV
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["self_battery-EV-charging"], name="EV2EV_self-consump.", stackgroup='negative',line={'color':'#fff824'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            # EV给电网
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["battery-grid-export"], name="EV2grid_export", stackgroup='negative'),
            row=1, col=1, secondary_y=False)
        #产生的能源 
        #Production sources
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["solar_power"], name="PRO_solar_power",line={'color':'#95bf00'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["wind_power"], name="PRO_wind_power",line={'color':'#1ac6ff'}, stackgroup='produced'),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=-self.VPP_loads["RE_power"], name="PRO_RE_power",line={'color':'rgb(45, 167, 176)'}),
            row=1, col=1, secondary_y=False)
        VPP_opt_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.VPP_loads["discharging_ev_power"],line={'color':'#fa1d9c'}, name="PRO_ev_discharged_pwr", stackgroup='produced'),
            row=1, col=1, secondary_y=False)


        VPP_opt_fig['layout']['yaxis1'].update(title='kW')
        #VPP_opt_fig['layout']['yaxis2'].update(title='kW')
        VPP_opt_fig['layout']['legend'].update(title='Time series')
        VPP_opt_fig.update_layout(width=1500,height= 750,
                                    #barmode='stack', 
                                    title_text='VPP Supply/demand power',
                                    xaxis_rangeslider_visible=True, xaxis_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-06-10 00:00:00"])
        #VPP_opt_fig.show()
        return VPP_opt_fig

    def plot_rewards_results(self):
        """
        在有控制的充电行动的VPP仿真中，奖励(每一步的total load，离开时电动车剩余电量)随时间推移的变化。
        Method to plot and visualize the rewards (total load for every step, EVs energy left at departure)
        over time during the VPP simulation with controlled charging actions.
        """
        rewards_fig = make_subplots(subplot_titles=('Load reward over time','EVs reward over time'),
                            rows=2, cols=1, shared_xaxes=True,
                            specs=[[{"secondary_y": True}],
                                    [{"secondary_y": True}]])
        
        rewards_serie = self.optimized_VPP_data["rewards"].values
        rewards_serie[-2] = 0
        # Top graph
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.optimized_VPP_data["total_load"], line={'color':'#9467bd'}, name="total_load", stackgroup='load'),
            row=1, col=1, secondary_y=False)
        rewards_fig.add_trace(
            go.Scatter(x=self.elvis_time_serie, y=self.load_reward_hist, line={'color':'rgb(45, 167, 176)'}, name="load_rewards", stackgroup='reward'),
            row=1, col=1, secondary_y=True)
        
        rewards_fig.add_trace(
            # 可用EV数=充电站数-空闲充电站数
            go.Scatter(x=self.elvis_time_serie, y=self.avail_EVs_n, line={'color':'rgb(238, 173, 81)'}, name="EVs_available", stackgroup='evs'),
            row=2, col=1, secondary_y=False)
        rewards_fig.add_trace(
            # 就是self.optimized_VPP_data["rewards"] 即self.reward_hist,包括EV离开车内剩余能源奖励和负载是否为0(EV正好能满足用户和其他EV)的奖励、最后一步在仿真结束时评估最终的agent奖励
            go.Scatter(x=self.elvis_time_serie, y=rewards_serie, line={'color':'rgb(115, 212, 127)'}, name="total_reward", stackgroup='tot_reward'),
            row=2, col=1, secondary_y=True)
        rewards_fig.add_trace(
            # EVs_energy_leaving_reward_t  # 执行情况的奖励记录到上一step的hist历史中
            go.Scatter(x=self.elvis_time_serie, y=self.EVs_reward_hist, line={'color':'rgb(210, 80, 75)'}, name="EVs_rewards"),
            row=2, col=1, secondary_y=True)

        self.load_reward_hist
        self.VPP_table["EV_reward"] = self.EVs_reward_hist
        self.VPP_table["rewards"] = self.reward_hist
        rewards_fig['layout']['yaxis1'].update(title='kW')
        rewards_fig['layout']['yaxis2'].update(title='Score')
        rewards_fig['layout']['yaxis3'].update(title='n_EVs')
        rewards_fig['layout']['yaxis4'].update(title='Score')
        
        rewards_fig.update_layout(title_text='Rewards results', width=1500,height=600, xaxis2_rangeslider_visible=True, xaxis2_rangeslider_thickness=0.05, xaxis_range=["2022-06-01 00:00:00", "2022-07-01 00:00:00"])
        return rewards_fig

    def plot_rewards_stats(self):
        """
        # 绘制奖励统计
        VPP agent累计奖励和构成它的每个奖励实例按类别划分。cumulative = final_reward + step_total, step_total = step_EV_en + step_load
        - cumulative累计：所有奖励实例的总和（最终奖励和步骤奖励）。 =np.sum(self.reward_hist)
        - final_total：最终奖励4个实例的总和:EV离开时平均剩余能量的奖励、Load>0的负载奖励买电越多越惩罚、未消纳的风光越多越惩罚、只买电的总消费求和越多越惩罚
        - step_total: 阶梯奖励实例的总和:EV每次离开能量奖励sum(self.EVs_reward_hist) + 负载累积奖励越靠近0越高,即下面两个相加
        - step_EV_en：当电动汽车离开充电站时，根据剩余能量在时间步数t上给予的所有奖励的总和。
        - step_load：根据总负载在每个时间段给予的所有奖励之和。
        - final_Av_EV_en：在离开充电站时，对电动车平均剩余能量评估的最终奖励。
        - final_over_en：对从电网中消耗的总能量进行评估的最终奖励（非自给自足的）
        - final_under_en：对浪费的总能量（产生但未使用）进行评估的最终奖励。
        - final_overcost：对总的能源消耗价格（购买的能源）进行评估的最终奖励。
        
        Method to plot and visualize the VPP agent cumulative reward and each reward instance composing it divided per category:
        - cumulative: the sum of all reward instances (final-rewards and step-rewards)
        - final_total: the sum of the final-reward instances
        - step_total: the sum of the step-reward instances
        - step_EV_en: the sum of all the rewards given at timesteps t when an EV left the charging station according to the energy left
        - step_load: the sum of all the rewards given at each timestep according the total load
        - final_Av_EV_en: the final reward evaluated for the Average EVs energy left when leaving the charging station
        - final_over_en: the final reward evaluated for the total energy consumed from the grid (not autosufficient)
        - final_under_en: the final reward evaluated for the total energy wasted (produced but not used)
        - final_overcost: the final reward evaluated for the total energy consumed price (energy bought)
        """
        rewards_fig = make_subplots(rows=1, cols=1, specs=[[{"secondary_y": False}]])

        final_reward = (self.AV_EVs_energy_reward + self.overconsume_reward + self.underconsume_reward + self.overcost_reward)
        step_reward = (self.EVs_energy_reward + self.load_t_reward)
        rewards_fig.add_trace(go.Bar(x=["cumulative", 'final_total', "step_total", "step_EV_en", "step_load", "final_Av_EV_en", "final_Grid_en",  "final_RE-to-EV_unused_en","final_Grid-cost"],
                            y=[self.cumulative_reward, final_reward, step_reward, self.EVs_energy_reward, self.load_t_reward, self.AV_EVs_energy_reward, self.overconsume_reward, self.underconsume_reward, self.overcost_reward],
                            marker_color=['rgb(117, 122, 178)', 'rgb(156, 99, 255)', 'rgb(115, 212, 127)', 'rgb(210, 80, 75)', 'rgb(45, 167, 176)', 'rgb(238, 173, 81)', 'rgb(249, 152, 179)', 'rgb(77, 218, 193)', 'rgb(97, 159, 210)']),
                            row=1, col=1, secondary_y=False)
        
        #rewards_fig['layout']['yaxis1'].update(title='Score')
        rewards_fig['layout']['yaxis1'].update(title='Score')
        #rewards_fig['layout']['legend'].update(title=f'Cumulat.Reward= {round(self.cumulative_reward,2)}')
        rewards_fig.update_layout(title_text="Cumulative, Step, Final reward bars comparison", width=1500,height=500,)
        #rewards_fig.show()
        return rewards_fig
    
    def plot_VPP_Elvis_comparison(self):
        """
        用条形图显示VPP仿真的受控充电结果与ELVIS非受控充电结果相比较。
        Method to plot and visualize with bars the VPP simulation with controlled charging results compared to the ELVIS uncontrolled charging ones.
        """
        comparison_fig = make_subplots(subplot_titles=('Av.EVs energy at departure','Grid used en.','RE-to-vehicle unused en.', 'Grid-cost'),
                            rows=1, cols=4,
                            specs=[[{"secondary_y": False}, {"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])

        x = ["Elvis_simulation","VPP_simulation"]
        marker_color = ['#636efa', 'rgb(77, 218, 193)']
        comparison_fig.add_trace(go.Bar(x=["Elvis_simulation","VPP_simulation","Expected"], y=[self.Elvis_av_EV_energy_left, self.av_EV_energy_left, self.exp_ev_en_left], marker_color=['#636efa', 'rgb(77, 218, 193)','orange']),row=1, col=1)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_overconsume, self.overconsumed_en], marker_color=marker_color),row=1, col=2)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_underconsume, self.underconsumed_en], marker_color=marker_color),row=1, col=3)
        comparison_fig.add_trace(go.Bar(x=x, y=[self.Elvis_overcost, self.sim_overcost], marker_color=marker_color),row=1, col=4)
        comparison_fig['layout']['yaxis1'].update(title='kWh')
        comparison_fig['layout']['yaxis2'].update(title='kWh')
        comparison_fig['layout']['yaxis3'].update(title='kWh')
        comparison_fig['layout']['yaxis4'].update(title='€')
        comparison_fig.update_layout(title_text='VPP/Elvis simulation comparison', width=1500,height=500, showlegend = False)
        #comparison_fig.show()
        return comparison_fig

    def plot_EVs_arrival_en(self):
        """
        Method to plot and visualize the histogram of the EVs energy at arrival.
        """
        kpi_fig = px.histogram(x=self.EVs_energy_at_arrival, marginal = 'violin',
                                opacity=0.8,
                                color_discrete_sequence=['indianred'] # color of histogram bars
                                )
        kpi_fig.update_xaxes(title = 'energy% available (kWh)')
        kpi_fig.update_layout(title_text="EVs energy at arrival histogram",  width=1500,height=700,)
        #kpi_fig.show()
        return kpi_fig

    def plot_EVs_kpi(self):
        """
        在VPP仿真中，EV离开时能量区间计数图。
        Method to plot and visualize the histogram of the EVs energy left at departure during the VPP simulation with controlled charging.
        """
        kpi_fig = px.histogram(x=self.EVs_energy_at_leaving, marginal = 'violin')
        kpi_fig.update_xaxes(title = 'energy% left (kWh)')
        kpi_fig.update_layout(title_text="EVs energy at departure histogram",  width=1500,height=700,)
        #kpi_fig.show()
        return kpi_fig
    
    def plot_actions_kpi(self):
        """
        在VPP仿真中，agent所采取行动是否有效的热图。action_truth_table
        Method to plot and visualize the heatmap of the valid actions taken by the agent during the VPP simulation with controlled charging.
        """
        kpi_fig = make_subplots(subplot_titles=("Valid actions per station table","Vehicle availability per station table"),
                            rows=1, cols=2,
                            specs=[[{"secondary_y": False},{"secondary_y": False}]])

        kpi_fig.add_trace(go.Heatmap(z=self.action_truth_table.astype(int), colorscale='Viridis', colorbar_x=0.45), row=1, col=1)
        #self.Evs_id_table = self.Evs_id_table.astype(bool)
        kpi_fig.add_trace(go.Heatmap(z=self.Evs_id_table), row=1, col=2)
        kpi_fig.update_layout(title_text='Actions KPIs', width=1500,height=500,)
        kpi_fig['layout']['yaxis1'].update(title='timesteps')
        kpi_fig['layout']['yaxis2'].update(title='timesteps')
        kpi_fig['layout']['xaxis1'].update(title='ch.station number')
        kpi_fig['layout']['xaxis2'].update(title='ch.station number')
        #kpi_fig.show()
        return kpi_fig

    def plot_load_kpi(self):
        """
        # 取一周2022-01-01~2022-01-08、一月2022-06-01~2022-07-01、一年的负载数据柱状图分析, ELVIS和VPP对比
        Method to plot and visualize the histogram of the timesteps load values during the ELVIS uncontrolled charging simulation,
        and during the VPP simulation with controlled charging (Weekly, Monthly, Yearly)
        """
        kpi_fig = make_subplots(subplot_titles=('Elvis Weekly',"Elvis Monthly","Elvis Yearly",'VPP Weekly',"VPP Monthly","VPP Yearly"), rows=2, cols=3,
                            specs=[[{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}],
                                    [{"secondary_y": False},{"secondary_y": False},{"secondary_y": False}]])

        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"].loc["2022-01-01 00:00:00":"2022-01-08 00:00:00"], marker = dict(color ='#7663fa')), row=1, col=1)
        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"].loc["2022-06-01 00:00:00":"2022-07-01 00:00:00"], marker = dict(color ='#636efa')), row=1, col=2)
        kpi_fig.add_trace(go.Histogram(x=self.VPP_data["total_load"], marker = dict(color ='#636efa')), row=1, col=3)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"].loc["2022-01-01 00:00:00":"2022-01-08 00:00:00"], marker = dict(color ='#00cc96')), row=2, col=1)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"].loc["2022-06-01 00:00:00":"2022-07-01 00:00:00"], marker = dict(color ='#00cc96')), row=2, col=2)
        kpi_fig.add_trace(go.Histogram(x=self.optimized_VPP_data["total_load"], marker = dict(color ='rgb(77, 218, 193)')), row=2, col=3)
        
        kpi_fig['layout']['xaxis1'].update(title='kW')
        kpi_fig['layout']['xaxis2'].update(title='kW')
        kpi_fig['layout']['xaxis3'].update(title='kW')
        kpi_fig['layout']['xaxis4'].update(title='kW')
        kpi_fig['layout']['xaxis5'].update(title='kW')
        kpi_fig['layout']['xaxis6'].update(title='kW')
        kpi_fig['layout']['yaxis1'].update(title='load value occurences')
        kpi_fig['layout']['yaxis2'].update(title='load value occurences')
        kpi_fig['layout']['yaxis3'].update(title='load value occurences')
        kpi_fig['layout']['yaxis4'].update(title='load value occurences')
        kpi_fig['layout']['yaxis5'].update(title='load value occurences')
        kpi_fig['layout']['yaxis6'].update(title='load value occurences')
        kpi_fig.update_layout(title_text='Load peak occurences histograms',  width=1500,height=800, showlegend = False)
        #kpi_fig.show()
        return kpi_fig

    
    def plot_yearly_load_log(self):
        """
        按负载功率0.2kW为区间长度对step计数(年,叠加),画对数直方图,对比ELVIS和VPP.
        Method to plot and visualize the logaritmic histogram of the timesteps load values during the ELVIS uncontrolled charging simulation,
        during the VPP simulation with controlled charging (Yearly, superimposed)
        """
        # 35041个0
        x0 = [0]*self.tot_simulation_len
        # 35041个VPP未优化的负载
        x1 = self.VPP_data["total_load"].values
        # 计数total_load在-0.1 < i < 0.1的step个数(ELVIS)
        Elvis_zero_load_n = sum(1 for i in x1 if -0.1 < i < 0.1)
        # 2.07%
        time_zero_Elvis= (Elvis_zero_load_n/self.tot_simulation_len)*100
        x2 = self.optimized_VPP_data["total_load"].values
        # 计数total_load在-0.1 < i < 0.1的step个数(VPP)
        VPP_zero_load_n = sum(1 for i in x2 if -0.1 < i < 0.1)
        # 41.89%
        time_zero_VPP= (VPP_zero_load_n/self.tot_simulation_len)*100
        
        # concatenate组合成新的矩阵[3*35041行,2列],两列分别为名称series和值kW,即[35041个"steady-zero-load"对应值0, 35041个"ELVIS-load"对应值ELVIS负载,35041个"VPP-load"对应值值VPP负载]
        df =pd.DataFrame(dict(series = np.concatenate((["steady-zero-load"]*len(x0), ["ELVIS-load"]*len(x1), ["VPP-load"]*len(x2))), 
                                kW  = np.concatenate((x0,x1,x2))
                            ))

        #f"(-0.05)<load<(0.05)[kWh]:{VPP_zero_load_n}-steps":'orange', f"{self.EVs_n}EVs-ELVIS-load(zero:{round(time_zero_Elvis,1)}%)":'#636efa', f"{self.EVs_n}EVs-VPP-load(zero:{round(time_zero_VPP,1)}%)":'rgb(77, 218, 193)'
        # 对数直方图
        kpi_fig = px.histogram(df, x="kW", color="series", barmode="overlay", marginal = 'violin', log_y=True, color_discrete_map = {"steady-zero-load":'orange', "ELVIS-load":'#636efa', "VPP-load":'rgb(77, 218, 193)'})
        kpi_fig.update_layout(#title_text= f"{self.EVs_n}EVs-yearly-load-distr.:", 
                                title=dict(text= f"[{self.EVs_n}EVs weekly] Yearly-load-distribution.  Load in ±0.1 kW range  -VPP: {VPP_zero_load_n} steps ({round(time_zero_VPP,1)}%),  -Elvis: {Elvis_zero_load_n} steps ({round(time_zero_Elvis,1)}%)", 
                                            x=0.1, y=0.95,
                                            #font_family="Open Sherif",
                                            font_size=14,
                                            #font_color="red"
                                            ),
                                width=1500,height=700,
                                xaxis={'side': 'top'},)
        #kpi_fig.show()
        return kpi_fig