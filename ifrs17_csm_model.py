"""
============================================================================
Contractual Service Margin Calculation Model - Building Block Approach
============================================================================

Author: Judy Chang
Objective:實現 IFRS 17 準則下的 CSM 計算，並同時支援 TW-ICS 資本適足性評估
============================================================================
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Tuple

# 設定視覺化風格
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial Unicode MS', 'SimHei']
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")


class IFRS17_CSM_Model:
    """IFRS 17 CSM 計算模型"""
    
    def __init__(self, policy_data: Dict, assumptions: Dict, proj_years: int):
        """
        初始化模型
        
        Parameters:
        -----------
        policy_data : dict
            保單資料
        assumptions : dict
            精算與經濟假設
        proj_years : int
            預測年期
        """
        self.policy_data = policy_data
        self.assumptions = assumptions
        self.proj_years = proj_years
        
    def project_cashflows(self) -> pd.DataFrame:
        """
        模組 1: 預測履行現金流量
        
        Returns:
        --------
        pd.DataFrame
            包含逐年現金流量預測的資料框
        """
        # 初始化資料框
        cf = pd.DataFrame({
            'year': range(self.proj_years + 1),
            'policies_start_of_year': 0.0,
            'premium_inflow': 0.0,
            'death_outflow': 0.0,
            'surrender_outflow': 0.0,
            'expense_outflow': 0.0,
            'net_cashflow': 0.0,
            'policies_end_of_year': 0.0
        })
        
        # 設定初始狀態
        cf.loc[0, 'policies_end_of_year'] = self.policy_data['initial_policies']
        
        # 逐年預測
        for t in range(1, self.proj_years + 1):
            # 期初保單數
            policies_boy = cf.loc[t-1, 'policies_end_of_year']
            cf.loc[t, 'policies_start_of_year'] = policies_boy
            
            if policies_boy == 0:
                break
            
            # 取得假設
            mortality_rate = self.assumptions['mortality_rate'][t-1]
            surrender_rate = self.assumptions['surrender_rate'][t-1]
            
            # 保費收入
            prem_inflow = policies_boy * self.policy_data['premium_per_policy']
            cf.loc[t, 'premium_inflow'] = prem_inflow
            
            # 死亡給付
            death_claims = policies_boy * mortality_rate * self.policy_data['sum_assured_per_policy']
            cf.loc[t, 'death_outflow'] = death_claims
            
            # 脫退給付
            surrender_payouts = (policies_boy * (1 - mortality_rate) * surrender_rate * 
                               self.policy_data['cash_value_per_policy'][t-1])
            cf.loc[t, 'surrender_outflow'] = surrender_payouts
            
            # 費用支出
            acquisition_expense = self.assumptions['acq_expense_per_policy'] * policies_boy if t == 1 else 0
            maintenance_expense = policies_boy * self.assumptions['maint_expense_per_policy']
            commission_expense = prem_inflow * self.assumptions['commission_rate'][t-1]
            total_expense = acquisition_expense + maintenance_expense + commission_expense
            cf.loc[t, 'expense_outflow'] = total_expense
            
            # 淨現金流量（從公司角度：流入為正，流出為負）
            cf.loc[t, 'net_cashflow'] = prem_inflow - death_claims - surrender_payouts - total_expense
            
            # 期末保單數
            policies_eoy = policies_boy * (1 - mortality_rate) * (1 - surrender_rate)
            cf.loc[t, 'policies_end_of_year'] = policies_eoy
        
        return cf
    
    def calculate_discount_factors(self) -> np.ndarray:
        """
        模組 2: 計算折現因子
        
        Returns:
        --------
        np.ndarray
            折現因子陣列
        """
        spot_rates = np.array(self.assumptions['discount_curve'][:self.proj_years])
        years = np.arange(1, self.proj_years + 1)
        discount_factors = 1 / (1 + spot_rates) ** years
        return discount_factors
    
    def determine_risk_adjustment(self, cf: pd.DataFrame, df: np.ndarray) -> Dict:
        """
        模組 3: 計算風險調整（資本成本法）
        
        Parameters:
        -----------
        cf : pd.DataFrame
            現金流量預測
        df : np.ndarray
            折現因子
            
        Returns:
        --------
        dict
            包含風險調整總額與所需資本預測
        """
        required_capital = np.zeros(self.proj_years)
        
        for t in range(self.proj_years):
            # 未來現金流出
            future_outflows = (cf.loc[t+1:self.proj_years, 'death_outflow'].values +
                             cf.loc[t+1:self.proj_years, 'surrender_outflow'].values +
                             cf.loc[t+1:self.proj_years, 'expense_outflow'].values)
            
            if len(future_outflows) > 0 and t < len(df):
                # 相對折現因子
                df_relative = df[t:self.proj_years] / df[t]
                
                # 確保維度匹配
                min_len = min(len(future_outflows), len(df_relative)-1)
                if min_len > 0:
                    # 未來流出現值
                    pv_future_outflows = np.sum(future_outflows[:min_len] * df_relative[1:min_len+1])
                    
                    # 所需資本
                    required_capital[t] = pv_future_outflows * self.assumptions['capital_pct']
        
        # 資本成本流
        cost_of_capital = required_capital * self.assumptions['coc_rate']
        
        # 風險調整
        ra = np.sum(cost_of_capital * df)
        
        return {
            'RA': ra,
            'RC_Projection': required_capital
        }
    
    def calculate_BBA_components(self) -> Dict:
        """
        模組 4: 計算 BBA 各項組件並推導 CSM
        
        Returns:
        --------
        dict
            包含 PVFCF, RA, CSM, LRC 等關鍵指標
        """
        # 步驟 1: 預測現金流量
        cf = self.project_cashflows()
        
        # 步驟 2: 計算折現因子
        df = self.calculate_discount_factors()
        
        # 步驟 3: 計算履行現金流量現值
        # PVFCF 從負債角度：負數表示預期淨流入（盈利），正數表示預期淨流出（虧損）
        net_cfs = cf.loc[1:self.proj_years, 'net_cashflow'].values
        pvfcf = -np.sum(net_cfs * df)  # 負號轉換為負債視角
        
        # 步驟 4: 計算風險調整
        ra_result = self.determine_risk_adjustment(cf, df)
        ra = ra_result['RA']
        
        # 步驟 5: 計算 CSM
        fulfilment_value = pvfcf + ra
        
        if fulfilment_value > 0:
            # 虧損性合約
            csm = 0
            day_one_loss = fulfilment_value
        else:
            # 盈利性合約
            csm = -fulfilment_value
            day_one_loss = 0
        
        return {
            'PVFCF': pvfcf,
            'RA': ra,
            'CSM': csm,
            'Day1_Loss': day_one_loss,
            'LRC': pvfcf + ra + csm,
            'CashflowProjection': cf,
            'DiscountFactors': df
        }
    
    def amortize_csm(self, initial_csm: float, cf: pd.DataFrame) -> pd.DataFrame:
        """
        模組 5: CSM 攤銷計算
        
        Parameters:
        -----------
        initial_csm : float
            初始 CSM 金額
        cf : pd.DataFrame
            現金流量預測
            
        Returns:
        --------
        pd.DataFrame
            CSM 攤銷時間表
        """
        schedule = pd.DataFrame({
            'year': range(1, self.proj_years + 1),
            'csm_opening': 0.0,
            'interest_accretion': 0.0,
            'csm_release': 0.0,
            'csm_closing': 0.0
        })
        
        # 保障單位
        coverage_units = (cf.loc[1:self.proj_years, 'policies_start_of_year'].values * 
                         self.policy_data['sum_assured_per_policy'])
        
        # 鎖定利率
        locked_in_rate = self.assumptions['discount_curve'][0]
        
        # 設定初始 CSM
        schedule.loc[0, 'csm_opening'] = initial_csm
        
        # 逐年攤銷
        for t in range(self.proj_years):
            # 利息增值
            interest = schedule.loc[t, 'csm_opening'] * locked_in_rate
            schedule.loc[t, 'interest_accretion'] = interest
            
            # 攤銷金額
            csm_after_interest = schedule.loc[t, 'csm_opening'] + interest
            remaining_coverage = np.sum(coverage_units[t:])
            
            if remaining_coverage > 0:
                release = (coverage_units[t] / remaining_coverage) * csm_after_interest
            else:
                release = 0
            schedule.loc[t, 'csm_release'] = release
            
            # 期末 CSM
            closing = csm_after_interest - release
            schedule.loc[t, 'csm_closing'] = closing
            
            # 下期期初
            if t < self.proj_years - 1:
                schedule.loc[t + 1, 'csm_opening'] = closing
        
        return schedule
    
    def print_summary(self, results: Dict):
        """列印計算結果摘要"""
        print("\n" + "=" * 70)
        print("IFRS 17 建築區塊法 (BBA) 計算結果摘要")
        print("=" * 70)
        print(f"履行現金流量現值 (PVFCF): NT$ {results['PVFCF']/1e6:,.1f}M")
        print(f"風險調整 (RA):            NT$ {results['RA']/1e6:,.1f}M")
        print(f"合約服務邊際 (CSM):       NT$ {results['CSM']/1e6:,.1f}M")
        print(f"剩餘保障負債 (LRC):       NT$ {results['LRC']/1e6:,.1f}M")
        if results['Day1_Loss'] > 0:
            print(f"\n⚠️  第一日虧損:            NT$ {results['Day1_Loss']/1e6:,.1f}M")
        print("=" * 70 + "\n")


def create_visualizations(model: IFRS17_CSM_Model, results: Dict, 
                         csm_schedule: pd.DataFrame, output_dir: str = 'output'):
    """
    建立視覺化圖表
    
    Parameters:
    -----------
    model : IFRS17_CSM_Model
        模型實例
    results : dict
        BBA 計算結果
    csm_schedule : pd.DataFrame
        CSM 攤銷時間表
    output_dir : str
        輸出目錄
    """
    import os
    os.makedirs(output_dir, exist_ok=True)
    
    # 圖表 1: CSM 攤銷趨勢
    plt.figure(figsize=(12, 6))
    plt.plot(csm_schedule['year'], csm_schedule['csm_closing']/1e6, 
             marker='o', linewidth=2, markersize=6, color='#0072B2')
    plt.fill_between(csm_schedule['year'], csm_schedule['csm_closing']/1e6, 
                     alpha=0.2, color='#0072B2')
    plt.xlabel('Policy Year', fontsize=12, fontweight='bold')
    plt.ylabel('CSM Closing Balance (NT$ Million)', fontsize=12, fontweight='bold')
    plt.title('Contractual Service Margin (CSM) Amortization Trend\n' + 
              'Systematic Release of Unearned Profit', 
              fontsize=14, fontweight='bold', pad=20)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/csm_amortization_trend.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 圖表 2: 現金流量結構
    cf = results['CashflowProjection']
    cf_viz = cf[cf['year'] > 0].copy()
    
    plt.figure(figsize=(12, 6))
    plt.stackplot(cf_viz['year'],
                  cf_viz['premium_inflow']/1e6,
                  cf_viz['death_outflow']/1e6,
                  cf_viz['surrender_outflow']/1e6,
                  cf_viz['expense_outflow']/1e6,
                  labels=['Premium Inflow', 'Death Benefits', 
                         'Surrender Benefits', 'Expenses'],
                  colors=['#00BA38', '#F8766D', '#619CFF', '#FF6B9D'],
                  alpha=0.8)
    plt.xlabel('Policy Year', fontsize=12, fontweight='bold')
    plt.ylabel('Amount (NT$ Million)', fontsize=12, fontweight='bold')
    plt.title('Fulfilment Cashflow Structure Analysis\n' +
              '20-Year Cashflow Composition', 
              fontsize=14, fontweight='bold', pad=20)
    plt.legend(loc='upper right', framealpha=0.9)
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    plt.savefig(f'{output_dir}/cashflow_structure.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    # 圖表 3: BBA 組件分解
    components = pd.DataFrame({
        'Component': ['PVFCF', 'Risk Adjustment', 'CSM'],
        'Amount': [results['PVFCF']/1e6, results['RA']/1e6, results['CSM']/1e6]
    })
    
    plt.figure(figsize=(10, 6))
    colors = ['#E74C3C' if x < 0 else '#27AE60' for x in components['Amount']]
    bars = plt.barh(components['Component'], components['Amount'], color=colors, alpha=0.8)
    plt.xlabel('Amount (NT$ Million)', fontsize=12, fontweight='bold')
    plt.title('IFRS 17 Building Block Approach (BBA) Components\n' +
              'Liability Composition at Initial Recognition', 
              fontsize=14, fontweight='bold', pad=20)
    plt.axvline(x=0, color='black', linestyle='-', linewidth=0.8)
    for i, v in enumerate(components['Amount']):
        plt.text(v + 10 if v > 0 else v - 10, i, f'NT$ {abs(v):.1f}M', 
                va='center', ha='left' if v > 0 else 'right', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig(f'{output_dir}/bba_components.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print(f"所有圖表已儲存至 {output_dir}/ 目錄")


# 主程式執行範例
if __name__ == "__main__":
    
    print("\n" + "="*70)
    print("IFRS 17 CSM 計算模型 - 執行範例")
    print("="*70 + "\n")
    
    # 設定保單資料
    policy_data = {
        'initial_policies': 10000,
        'premium_per_policy': 18000,  # 提高保費
        'sum_assured_per_policy': 1000000,
        'cash_value_per_policy': [0, 8000, 16000, 24000, 32000,
                                  40000, 48000, 56000, 64000, 72000,
                                  80000, 88000, 96000, 104000, 112000,
                                  120000, 128000, 136000, 144000, 150000],
        'policy_term': 20
    }
    
    # 設定精算假設
    proj_years = 20
    assumptions = {
        'mortality_rate': [0.00050, 0.00052, 0.00055, 0.00058, 0.00062,
                          0.00067, 0.00073, 0.00080, 0.00088, 0.00098,
                          0.00110, 0.00124, 0.00141, 0.00161, 0.00185,
                          0.00214, 0.00249, 0.00291, 0.00341, 0.00401],
        'surrender_rate': [0.05, 0.08, 0.10, 0.12, 0.10,
                          0.08, 0.06, 0.05, 0.04, 0.03,
                          0.03, 0.02, 0.02, 0.02, 0.02,
                          0.01, 0.01, 0.01, 0.01, 0.01],
        'acq_expense_per_policy': 6000,  # 降低首期費用
        'maint_expense_per_policy': 400,  # 降低維持費用
        'commission_rate': [0.30, 0.04, 0.02, 0.02, 0.01,  # 降低佣金
                           0.01, 0.01, 0.01, 0.01, 0.01,
                           0.00, 0.00, 0.00, 0.00, 0.00,
                           0.00, 0.00, 0.00, 0.00, 0.00],
        'discount_curve': [0.020] * proj_years,  # 提高折現率
        'capital_pct': 0.06,  # 降低資本要求
        'coc_rate': 0.05  # 降低資本成本
    }
    
    # 建立模型並執行計算
    print("正在執行 IFRS 17 BBA 計算...")
    model = IFRS17_CSM_Model(policy_data, assumptions, proj_years)
    results = model.calculate_BBA_components()
    model.print_summary(results)
    
    # 執行 CSM 攤銷
    print("正在計算 CSM 攤銷時間表...")
    csm_schedule = model.amortize_csm(results['CSM'], results['CashflowProjection'])
    print("\nCSM 攤銷時間表（前 5 年）:")
    print(csm_schedule.head().to_string(index=False))
    print()
    
    # 生成視覺化圖表
    print("\n正在生成視覺化圖表...")
    create_visualizations(model, results, csm_schedule)
    
    # 敏感性分析
    print("\n正在執行敏感性分析...")
    scenarios = {
        'Baseline': lambda a: a,
        'Mortality +10%': lambda a: {**a, 'mortality_rate': [x*1.10 for x in a['mortality_rate']]},
        'Lapse +20%': lambda a: {**a, 'surrender_rate': [x*1.20 for x in a['surrender_rate']]},
        'Discount -50bps': lambda a: {**a, 'discount_curve': [x-0.005 for x in a['discount_curve']]}
    }
    
    sensitivity_results = []
    for name, modifier in scenarios.items():
        modified_assumptions = modifier(assumptions.copy())
        scenario_model = IFRS17_CSM_Model(policy_data, modified_assumptions, proj_years)
        scenario_results = scenario_model.calculate_BBA_components()
        sensitivity_results.append({
            'Scenario': name,
            'PVFCF': scenario_results['PVFCF']/1e6,
            'RA': scenario_results['RA']/1e6,
            'CSM': scenario_results['CSM']/1e6
        })
    
    sensitivity_df = pd.DataFrame(sensitivity_results)
    baseline_csm = sensitivity_df.loc[0, 'CSM']
    sensitivity_df['CSM_Change_%'] = (sensitivity_df['CSM'] - baseline_csm) / baseline_csm * 100
    
    print("\n敏感性分析結果:")
    print(sensitivity_df.to_string(index=False))
    
    # 敏感性分析圖表
    plt.figure(figsize=(10, 6))
    colors = ['#3498DB', '#E74C3C', '#F39C12', '#9B59B6']
    bars = plt.barh(sensitivity_df['Scenario'], sensitivity_df['CSM'], color=colors, alpha=0.8)
    plt.xlabel('CSM (NT$ Million)', fontsize=12, fontweight='bold')
    plt.title('CSM Sensitivity Analysis\nImpact of Different Assumption Scenarios', 
              fontsize=14, fontweight='bold', pad=20)
    for i, v in enumerate(sensitivity_df['CSM']):
        plt.text(v + 10, i, f'{v:.1f}M', va='center', fontweight='bold')
    plt.grid(True, alpha=0.3, axis='x')
    plt.tight_layout()
    plt.savefig('output/csm_sensitivity_analysis.png', dpi=300, bbox_inches='tight')
    plt.close()
    
    print("\n" + "="*70)
    print("執行完成！所有結果已儲存至 output/ 目錄")
    print("="*70 + "\n")
