from flask import Flask, jsonify, request
import pandas as pd
import numpy as np
import base64
from io import StringIO
from flask_cors import CORS

app = Flask(__name__)
CORS(app)  # Isso permitirá chamadas de qualquer origem (em desenvolvimento)

# Carregar dados
csvdmgp1 = pd.read_csv('esea_master_dmg_demos.part1.csv')
csvkillsp1 = pd.read_csv('esea_master_kills_demos.part1.csv')
csvdemosp1 = pd.read_csv('esea_meta_demos.part1.csv')
master_demos = pd.read_csv('mm_master_demos.csv', index_col=0)
map_data = pd.read_csv('map_data.csv', index_col=0)
grenades_demos = pd.read_csv('mm_grenades_demos.csv')

def dataLenMap(data, data_demos, x, maps, round_t):
    data_demos = data_demos[data_demos['map'].isin(maps)]
    data_demos = data_demos[data_demos['round_type'].isin(round_t)]

    allfiles = data_demos['file'].unique()
    data_demos = data_demos[data_demos['file'].isin(allfiles[:x])]

    demos_unique = data_demos['file'].unique()
    filtered_csvdata_XGames_XMaps = data[data['file'].isin(demos_unique)]
    return filtered_csvdata_XGames_XMaps


def wpdata(filtered_csvdmgp1):
    grouped_df_hp = filtered_csvdmgp1.groupby('wp')['hp_dmg'].sum().reset_index()
    grouped_df_hp = grouped_df_hp.sort_values(by='hp_dmg', ascending=False)
    total_hp_dmg = grouped_df_hp['hp_dmg'].sum()
    grouped_df_hp['percentagem_hp'] = (grouped_df_hp['hp_dmg'] / total_hp_dmg) * 100
    
    grouped_df_arm = filtered_csvdmgp1.groupby('wp')['arm_dmg'].sum().reset_index()
    grouped_df_arm = grouped_df_arm.sort_values(by='arm_dmg', ascending=False)
    total_arm_dmg = grouped_df_arm['arm_dmg'].sum()
    grouped_df_arm['percentagem_arm'] = (grouped_df_arm['arm_dmg'] / total_arm_dmg) * 100
    
    merged_df = pd.merge(grouped_df_arm, grouped_df_hp, on='wp', how='inner')
    merged_df['percentagem_total'] = merged_df['percentagem_arm'] + merged_df['percentagem_hp']
    filtered_df = merged_df[merged_df['percentagem_total'] >= 2]
    filtered_df = filtered_df.sort_values(by='percentagem_total', ascending=False)
    return filtered_df

# Endpoints
@app.route('/api/scatterPlotDt', methods=['GET'])
def scatterPlotDt():
    map_name = request.args.get('map', 'de_mirage')
    games_count = int(request.args.get('games_count', 5))

    symbol_map = { 'PISTOL_ROUND': 'circle', 'ECO': 'cross', 'NORMAL': 'square', 'FORCE_BUY': 'triangle', 'SEMI_ECO': 'diamond'}
    winner_map = {'Terrorist': '#ff5722','CounterTerrorist': '#2196F3'}

    file = csvdemosp1.copy()
    filtered_XMaps = file[file['map'].isin([map_name])]

    allfiles = filtered_XMaps['file'].unique()

    filtered_XMaps_XGames = filtered_XMaps[filtered_XMaps['file'].isin(allfiles[:games_count])]
    #print(filtered_XMaps_XGames)
    winner_val = []
    mean_val = []

    for i in range(len(filtered_XMaps_XGames)):
        if filtered_XMaps_XGames['winner_side'].iloc[i] == 'CounterTerrorist':
            winner_val.append(filtered_XMaps_XGames['ct_eq_val'].iloc[i])
        else:
            winner_val.append(filtered_XMaps_XGames['t_eq_val'].iloc[i])
        mean_val.append((filtered_XMaps_XGames['ct_eq_val'].iloc[i]+filtered_XMaps_XGames['t_eq_val'].iloc[i])/2)

    filtered_XMaps_XGames_2 = filtered_XMaps_XGames.copy()

    filtered_XMaps_XGames_2['winner_val'] = winner_val
    filtered_XMaps_XGames_2['mean_val'] = mean_val
    filtered_XMaps_XGames_2 = filtered_XMaps_XGames_2.sort_values(by='round')

    df_medias_por_ronda = (
        filtered_XMaps_XGames_2
        .groupby('round')
        .agg({
            'ct_eq_val': 'mean',
            't_eq_val': 'mean',
            'winner_val': 'mean',
            'mean_val': 'mean',
            'winner_side': lambda x: x.value_counts().idxmax(),
            'round_type': lambda x: x.value_counts().idxmax()
        })
        .reset_index()
    )

    numeric_cols = ['ct_eq_val', 't_eq_val', 'winner_val', 'mean_val']
    df_medias_por_ronda[numeric_cols] = df_medias_por_ronda[numeric_cols].round(2)

    df_medias_por_ronda['plotly_symbol'] = df_medias_por_ronda['round_type'].map(symbol_map)
    df_medias_por_ronda['plotly_colors'] = df_medias_por_ronda['winner_side'].map(winner_map)
    
    media_1_14 = df_medias_por_ronda[df_medias_por_ronda['round'].between(1, 14)]['mean_val'].mean()
    media_15_29 = df_medias_por_ronda[df_medias_por_ronda['round'].between(15, 29)]['mean_val'].mean()
    media_30_mais = df_medias_por_ronda[df_medias_por_ronda['round'] >= 30]['mean_val'].mean()
    max_rounds = df_medias_por_ronda['round'].max()

    return jsonify({
        'ct_eq_val': df_medias_por_ronda['ct_eq_val'].tolist(),
        't_eq_val': df_medias_por_ronda['t_eq_val'].tolist(),
        'mean_val': df_medias_por_ronda['mean_val'].tolist(),
        'rounds': df_medias_por_ronda['round'].tolist(),
        'symbols': df_medias_por_ronda['plotly_symbol'].tolist(),
        'colors': df_medias_por_ronda['plotly_colors'].tolist(),
        'medias': {
            '1_14': round(float(media_1_14)), 
            '15_29': round(float(media_15_29)),
            '30_mais': round(float(media_30_mais))
        },
        'max_round': int(max_rounds)
    })

@app.route('/api/weapon_usage', methods=['GET'])
def weapon_usage():
    map_name = request.args.get('map', 'de_mirage')
    games_count = int(request.args.get('games_count', 5))
    
    maps = ['de_mirage','de_dust2','de_nuke','de_cache','de_inferno','de_overpass','de_train']
    allfiles = csvdemosp1['file'].unique()
    
    filtered_csvdmgp1 = dataLenMap(csvdmgp1, csvdemosp1, games_count, [map_name], allfiles)
    weapon_data = wpdata(filtered_csvdmgp1)
    
    return jsonify({
        'weapons': weapon_data['wp'].tolist(),
        'damage_percentages': weapon_data['percentagem_total'].tolist()
    })

@app.route('/api/weapon-usage2', methods=['GET'])
def weapon_usage2():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    games_count = int(request.args.get('games_count', 5))
    
    filtered_csvdmgp1 = dataLenMap(csvdmgp1, csvdemosp1, games_count, [map_name], [round_type])
    weapon_data = wpdata(filtered_csvdmgp1)
    
    return jsonify({
        'weapons': weapon_data['wp'].tolist(),
        'hp_percentages': weapon_data['percentagem_hp'].tolist(),
        'arm_percentages': weapon_data['percentagem_arm'].tolist()
    })

@app.route('/api/filterVicTypeRound', methods=['GET'])
def filterVicTypeRound():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    games_count = int(request.args.get('games_count', 5))
    round_types = csvdemosp1['round_type'].unique().tolist()
    filtered_csvdemosp1 = dataLenMap(csvdemosp1, csvdemosp1, games_count, [map_name], round_types)

    t_wins = filtered_csvdemosp1[filtered_csvdemosp1['winner_side'] == 'Terrorist']
    ct_wins = filtered_csvdemosp1[filtered_csvdemosp1['winner_side'] == 'CounterTerrorist']
    round_type_counts_t = round(t_wins['round_type'].value_counts(normalize=True) * 100)
    round_type_counts_ct = round(ct_wins['round_type'].value_counts(normalize=True) * 100)
    common_categories = list(set(round_type_counts_t.index) | set(round_type_counts_ct.index))
    round_type_counts_t = round_type_counts_t.reindex(common_categories, fill_value=0)
    round_type_counts_ct = round_type_counts_ct.reindex(common_categories, fill_value=0)
    index = np.arange(len(common_categories))

    return jsonify({
        'index': index.tolist(), 
        'round_type_counts_t': round_type_counts_t.tolist(), 
        'round_type_counts_ct': round_type_counts_ct.tolist(), 
        'common_categories': common_categories
    })

@app.route('/api/perVicdata', methods=['GET'])
def perVicdata():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    games_count = int(request.args.get('games_count', 5))

    filtered_csvdmgp1 = dataLenMap(csvdmgp1, csvdemosp1, games_count, [map_name], [round_type])

    grouped_df = filtered_csvdmgp1['vic_side']
    counts = grouped_df.value_counts()
    total = counts.sum()
    ct_percent = (counts['CounterTerrorist'] / total) * 100
    t_percent = (counts['Terrorist'] / total) * 100

    return jsonify({
        'ct_percent': ct_percent, 
        't_percent': t_percent
    })

@app.route('/api/bombdata', methods=['GET'])
def bombdata():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    games_count = int(request.args.get('games_count', 5))

    filtered_killsp1 = dataLenMap(csvkillsp1, csvdemosp1, games_count, [map_name], [round_type])

    last_actions = filtered_killsp1.groupby(['file', 'round']).tail(1)
    terrorist_last_actions = last_actions[last_actions['vic_side'] == 'Terrorist']
    bomb_counts = terrorist_last_actions['is_bomb_planted'].value_counts()
    total_actions = bomb_counts.sum()
    bomb_percentages = round(((bomb_counts / total_actions) * 100),2)

    bomb_plantada = bomb_percentages.values[1]
    bomb_nao_plantada = bomb_percentages.values[0]
    
    return jsonify({
        'bomb_plantada': bomb_plantada,
        'bomb_nao_plantada': bomb_nao_plantada
    })

@app.route('/api/money_spending', methods=['GET'])
def money_spending():
    map_name = request.args.get('map', 'de_mirage')
    games_count = int(request.args.get('games_count', 5))

    file = csvdemosp1

    filtered_XMaps = file[file['map'].isin([map_name])]

    allfiles = filtered_XMaps['file'].unique()

    filtered_XMaps_XGames = filtered_XMaps[filtered_XMaps['file'].isin(allfiles[:games_count])]
    
    filtered_XMaps_XGames = filtered_XMaps_XGames.sort_values(by='round')
    
    df_medias_por_ronda = (
        filtered_XMaps_XGames
        .groupby('round')
        .agg({
            'ct_eq_val': 'mean',
            't_eq_val': 'mean'
        })
        .reset_index()
    )
    
    numeric_cols = ['ct_eq_val', 't_eq_val']
    df_medias_por_ronda[numeric_cols] = df_medias_por_ronda[numeric_cols].round(2)

    return jsonify({
        'rounds': df_medias_por_ronda['round'].tolist(),
        'ct_values': df_medias_por_ronda['ct_eq_val'].tolist(),
        't_values': df_medias_por_ronda['t_eq_val'].tolist()
    })

@app.route('/api/utilityDt', methods=['GET'])
def utilityDt():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    games_count = int(request.args.get('games_count', 5))
    file = grenades_demos

    filtered_XMaps = file[file['map'].isin([map_name])]
    filtered_XMaps_Rounds = filtered_XMaps[filtered_XMaps['round_type'].isin([round_type])]

    allfiles = filtered_XMaps_Rounds['file'].unique()

    filtered_XMaps_XGames = filtered_XMaps_Rounds[filtered_XMaps_Rounds['file'].isin(allfiles[:games_count])]

    data_u = filtered_XMaps_XGames.groupby('nade').size().reset_index(name='count')

    return jsonify({
        'data_u_nade': data_u['nade'].tolist(),
        'data_u_count': data_u['count'].tolist()
    })

def filterHeatMap(master_demos_aux, map_data, round_type, maps_mm, smap):
    master_demos_aux = master_demos_aux[master_demos_aux['map'].isin(maps_mm)]
    master_demos_aux = master_demos_aux.reset_index(drop=True)
    
    md = map_data.loc[master_demos_aux['map']]
    md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = (master_demos_aux.set_index('map')[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']])
    
    md['att_pos_x'] = (md['ResX']*(md['att_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
    md['att_pos_y'] = (md['ResY']*(md['att_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])
    md['vic_pos_x'] = (md['ResX']*(md['vic_pos_x']-md['StartX']))/(md['EndX']-md['StartX'])
    md['vic_pos_y'] = (md['ResY']*(md['vic_pos_y']-md['StartY']))/(md['EndY']-md['StartY'])

    master_demos_aux[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']] = md[['att_pos_x', 'att_pos_y', 'vic_pos_x', 'vic_pos_y']].values

    rounds = list(range(1, 31))
    df_aux = master_demos_aux[(master_demos_aux['round'].isin(rounds)) & (master_demos_aux['round_type']==(round_type))]

    plot_df_t = df_aux.loc[(df_aux['map'] == smap) & (df_aux['att_side'] == 'Terrorist')]
    plot_df_ct = df_aux.loc[(df_aux['map'] == smap) & (df_aux['att_side'] == 'CounterTerrorist')]

    with open(f'{smap}.png', 'rb') as image_file:
        encoded_image = base64.b64encode(image_file.read()).decode()
    bg_image_url = f"data:image/png;base64,{encoded_image}"

    return plot_df_t, plot_df_ct, bg_image_url


@app.route('/api/heatmap', methods=['GET'])
def heatmap():
    map_name = request.args.get('map', 'de_mirage')
    round_type = request.args.get('round_type', 'PISTOL_ROUND')
    side = request.args.get('side', 'T')
    
    maps_mm = ['de_mirage', 'de_dust2', 'de_cbble','de_cache','de_inferno','de_overpass','de_train']
    plot_df_t, plot_df_ct, bg_image_url = filterHeatMap(master_demos, map_data, round_type, maps_mm, map_name)
    
    if side == 'T':
        df = plot_df_t
    else:
        df = plot_df_ct
    
    return jsonify({
        'x': df['att_pos_x'].tolist(),
        'y': df['att_pos_y'].tolist(),
        'bg_image_url': bg_image_url
    })

@app.route('/api/fMetricSquares', methods=['GET'])
def fMetricSquares():
    maps = ['de_mirage','de_dust2','de_cache','de_inferno','de_overpass','de_train']
    # WIN RATE - csvdmgp1, csvdemosp1
    file = csvdemosp1
    filtered_XMaps = file[file['map'].isin(maps)]
    allfiles = filtered_XMaps['file'].unique()

    filtered_csvdmgp1 = csvdmgp1[csvdmgp1['file'].isin(allfiles)]
    grouped_df = filtered_csvdmgp1['vic_side']
    counts = grouped_df.value_counts()
    total = counts.sum()
    ct_percent_g = (counts['CounterTerrorist'] / total) * 100
    t_percent_g = (counts['Terrorist'] / total) * 100

    # BOMB PLANT RATE - csvkillsp1, csvdemosp1
    filtered_killsp1 = csvkillsp1[csvkillsp1['file'].isin(allfiles)]
    last_actions = filtered_killsp1.groupby(['file', 'round']).tail(1)
    terrorist_last_actions = last_actions[last_actions['vic_side'] == 'Terrorist']
    bomb_counts = terrorist_last_actions['is_bomb_planted'].value_counts()
    total_actions = bomb_counts.sum()
    bomb_percentages = round(((bomb_counts / total_actions) * 100),2)
    bomb_plantada_g = bomb_percentages.values[1]
    bomb_nao_plantada_g = bomb_percentages.values[0]

    filtered_XMaps_XGames = filtered_XMaps[filtered_XMaps['file'].isin(allfiles)]

    # ROUND TIME
    filtered_XMaps_XGames_AUX = filtered_XMaps_XGames.copy()
    filtered_XMaps_XGames_AUX['diff_segundos'] = filtered_XMaps_XGames_AUX['end_seconds'] - filtered_XMaps_XGames_AUX['start_seconds']
    def converter_para_minutos_segundos(segundos):
        minutos = int(segundos // 60)
        segundos_restantes = int(round(segundos % 60))
        return f"{minutos:02d}:{segundos_restantes:02d}"
    ct_mean_seconds = filtered_XMaps_XGames_AUX[filtered_XMaps_XGames_AUX['winner_side'] == 'CounterTerrorist']['diff_segundos'].mean()
    t_mean_seconds = filtered_XMaps_XGames_AUX[filtered_XMaps_XGames_AUX['winner_side'] == 'Terrorist']['diff_segundos'].mean()
    ct_time_g = converter_para_minutos_segundos(ct_mean_seconds)
    t_time_g = converter_para_minutos_segundos(t_mean_seconds)

    # AVG TEAM MONEY
    filtered_XMaps_XGames_AUX = filtered_XMaps_XGames_AUX.sort_values(by='round')
    df_medias_por_ronda = (
        filtered_XMaps_XGames_AUX
        .groupby('round')
        .agg({
            'ct_eq_val': 'mean',
            't_eq_val': 'mean'
        })
        .reset_index()
    )
    numeric_cols = ['ct_eq_val', 't_eq_val']
    df_medias_por_ronda[numeric_cols] = df_medias_por_ronda[numeric_cols].round(2)

    return jsonify({
        'ct_percent_g': ct_percent_g,
        't_percent_g': t_percent_g,
        'ct_time_g': ct_time_g,
        't_time_g': t_time_g,
        "ct_mean_seconds": ct_mean_seconds,
        "t_mean_seconds": t_mean_seconds,
        'bomb_plantada_g': bomb_plantada_g,
        'bomb_nao_plantada_g': bomb_nao_plantada_g,
        'ct_values_g': df_medias_por_ronda['ct_eq_val'].tolist(),
        't_values_g': df_medias_por_ronda['t_eq_val'].tolist()
    })



@app.route('/api/maps', methods=['GET'])
def get_maps():
    maps = ['de_mirage','de_dust2','de_cache','de_inferno','de_overpass','de_train']
    return jsonify({'maps': maps})

@app.route('/api/round-types', methods=['GET'])
def get_round_types():
    round_types = csvdemosp1['round_type'].unique().tolist()
    return jsonify({'round_types': round_types})

if __name__ == '__main__':
    app.run(debug=True, port=5000)