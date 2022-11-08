from energym.examples.Controller import LabController
import energym


def get_ApartTherm_kpis(lower_tol, upper_tol):
    weather = "ESP_CT_Barcelona"
    num_sim_days = 3
    env = energym.make(
        "ApartmentsThermal-v0", weather=weather, simulation_days=num_sim_days)

    inputs = env.get_inputs_names()
    controller = LabController(
        control_list=inputs, lower_tol=lower_tol, upper_tol=upper_tol,
        nighttime_setback=True, nighttime_start=18, nighttime_end=6,
        nighttime_temp=18)

    steps = 480 * num_sim_days
    out_list = []
    outputs = env.get_output()
    controls = []
    hour = 0
    for i in range(steps):
        control = controller.get_control(outputs, 21, hour)
        control['P1_T_Tank_sp'] = [40.0]
        control['P2_T_Tank_sp'] = [40.0]
        control['P3_T_Tank_sp'] = [40.0]
        control['P4_T_Tank_sp'] = [40.0]
        control['Bd_Ch_EVBat_sp'] = [1.0]
        control['Bd_DisCh_EVBat_sp'] = [0.0]
        control['HVAC_onoff_HP_sp'] = [1.0]
        control['Bd_T_HP_sp'] = [45]
        controls += [{p: control[p][0] for p in control}]
        outputs = env.step(control)
        _, hour, _, _ = env.get_date()
        out_list.append(outputs)

    kpis = env.get_kpi()
    energy = kpis['kpi1']['kpi']
    num_rooms = 8
    total_avg_dev = 0
    for k in range(num_rooms):
        avg_dev = kpis[f'kpi{k+2}']['kpi']
        total_avg_dev += avg_dev
    avg_avg_dev = total_avg_dev / 8
    env.close()
    return energy, avg_avg_dev
