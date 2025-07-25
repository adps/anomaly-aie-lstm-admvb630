#******************************************************************************
# Copyright (C) 2020-2022 Xilinx, Inc. All rights reserved.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#******************************************************************************

##################################################################
# DESIGN PROCs
##################################################################

# Procedure to create entire design; Provide argument to make
# procedure reusable. If parentCell is "", will use root.
proc create_root_design { parentCell } {

  variable script_folder
  variable design_name

  if { $parentCell eq "" } {
    set parentCell [get_bd_cells /]
  }

  # Get object for parentCell
  set parentObj [get_bd_cells $parentCell]
  if { $parentObj == "" } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2090 -severity "ERROR" "Unable to find parent cell <$parentCell>!"}
    return
  }

  # Make sure parentObj is hier blk
  set parentType [get_property TYPE $parentObj]
  if { $parentType ne "hier" } {
    catch {common::send_gid_msg -ssname BD::TCL -id 2091 -severity "ERROR" "Parent <$parentObj> has TYPE = <$parentType>. Expected to be <hier>."}
    return
  }

  # Save current instance; Restore later
  set oldCurInst [current_bd_instance .]

  # Set parent object as current
  current_bd_instance $parentObj

  # Create interface ports
  set ddr4_bank0 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:ddr4_rtl:1.0 ddr4_bank0 ]

  # set ch0_lpddr4_trip2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 ch0_lpddr4_trip2 ]

  # set ch0_lpddr4_trip3 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 ch0_lpddr4_trip3 ]

  # set ch1_lpddr4_trip1 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 ch1_lpddr4_trip1 ]

  # set ch1_lpddr4_trip2 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 ch1_lpddr4_trip2 ]

  # set ch1_lpddr4_trip3 [ create_bd_intf_port -mode Master -vlnv xilinx.com:interface:lpddr4_rtl:1.0 ch1_lpddr4_trip3 ]

  set ddr4_bank0_sys_clk [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 ddr4_bank0_sys_clk ]
  set_property -dict [ list \
   CONFIG.FREQ_HZ {266666667} \
   ] $ddr4_bank0_sys_clk

  # set lpddr4_clk2 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 lpddr4_clk2 ]
  # set_property -dict [ list \
  #  CONFIG.FREQ_HZ {200200200} \
  #  ] $lpddr4_clk2

  # set lpddr4_clk3 [ create_bd_intf_port -mode Slave -vlnv xilinx.com:interface:diff_clock_rtl:1.0 lpddr4_clk3 ]
  # set_property -dict [ list \
  #  CONFIG.FREQ_HZ {200200200} \
  #  ] $lpddr4_clk3  



  # Create instance: CIPS_0, and set properties
  set CIPS_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:versal_cips:3.4 CIPS_0 ]
  set_property -dict [list \
    CONFIG.CLOCK_MODE {Custom} \
    CONFIG.DDR_MEMORY_MODE {Custom} \
    CONFIG.PS_BOARD_INTERFACE {ps_pmc_fixed_io} \
    CONFIG.PS_PL_CONNECTIVITY_MODE {Custom} \
    CONFIG.PS_PMC_CONFIG { \
      CLOCK_MODE {Custom} \
      DDR_MEMORY_MODE {Custom} \
      DESIGN_MODE {1} \
      DEVICE_INTEGRITY_MODE {Sysmon temperature voltage and external IO monitoring} \
      PMC_CRP_PL0_REF_CTRL_FREQMHZ {99.999992} \
      PMC_MIO16 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO17 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO18 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO19 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO20 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO21 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO22 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_MIO23 {{AUX_IO 0} {DIRECTION inout} {DRIVE_STRENGTH 8mA} {OUTPUT_DATA default} {PULL pullup} {SCHMITT 0} {SLEW slow} {USAGE GPIO}} \
      PMC_QSPI_FBCLK {{ENABLE 1} {IO {PMC_MIO 6}}} \
      PMC_QSPI_PERIPHERAL_MODE {Dual Parallel} \
      PMC_QSPI_PERIPHERAL_DATA_MODE {x4} \
      PMC_QSPI_PERIPHERAL_ENABLE {1} \
      PMC_REF_CLK_FREQMHZ {33.3333} \
      PMC_SD1_PERIPHERAL {{CLK_100_SDR_OTAP_DLY 0x00} {CLK_200_SDR_OTAP_DLY 0x00} {CLK_50_DDR_ITAP_DLY 0x00} {CLK_50_DDR_OTAP_DLY 0x00} {CLK_50_SDR_ITAP_DLY 0x2C} {CLK_50_SDR_OTAP_DLY 0x4} {ENABLE 1} {IO {PMC_MIO 26 .. 36}}} \
      PMC_SD1_SLOT_TYPE {SD 2.0} \
      PMC_USE_PMC_NOC_AXI0 {1} \
      PS_BOARD_INTERFACE {ps_pmc_fixed_io} \
      PS_ENET0_MDIO {{ENABLE 1} {IO {PMC_MIO 50 .. 51}}} \
      PS_ENET0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 0 .. 11}}} \
      PS_CAN0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 14 .. 15}}} \
      PS_CAN1_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 24 .. 25}}} \
      PS_GEN_IPI0_ENABLE {1} \
      PS_GEN_IPI0_MASTER {A72} \
      PS_GEN_IPI1_ENABLE {1} \
      PS_GEN_IPI1_MASTER {A72} \
      PS_GEN_IPI2_ENABLE {1} \
      PS_GEN_IPI2_MASTER {A72} \
      PS_GEN_IPI3_ENABLE {1} \
      PS_GEN_IPI3_MASTER {A72} \
      PS_GEN_IPI4_ENABLE {1} \
      PS_GEN_IPI4_MASTER {A72} \
      PS_GEN_IPI5_ENABLE {1} \
      PS_GEN_IPI5_MASTER {A72} \
      PS_GEN_IPI6_ENABLE {1} \
      PS_GEN_IPI6_MASTER {A72} \
      PS_I2C0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 22 .. 23}}} \
      PS_IRQ_USAGE {{CH0 1} {CH1 0} {CH10 0} {CH11 0} {CH12 0} {CH13 0} {CH14 0} {CH15 0} {CH2 0} {CH3 0} {CH4 0} {CH5 0} {CH6 0} {CH7 0} {CH8 0} {CH9 0}} \
      PS_NUM_FABRIC_RESETS {1} \
      PS_PCIE_EP_RESET1_IO {PS_MIO 18} \
      PS_PCIE_EP_RESET2_IO {PS_MIO 19} \
      PS_PCIE_RESET {ENABLE 1} \
      PS_PL_CONNECTIVITY_MODE {Custom} \
      PS_TTC0_PERIPHERAL_ENABLE {1} \
      PS_UART0_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 16 .. 17}}} \
      PS_UART1_PERIPHERAL {{ENABLE 1} {IO {PS_MIO 12 .. 13}}} \
      PS_USE_FPD_AXI_NOC0 {1} \
      PS_USE_FPD_AXI_NOC1 {1} \
      PS_USE_FPD_CCI_NOC {1} \
      PS_USE_M_AXI_FPD {1} \
      PS_USE_NOC_LPD_AXI0 {1} \
      PS_USE_PMCPL_CLK0 {1} \
      SMON_ALARMS {Set_Alarms_On} \
      SMON_ENABLE_TEMP_AVERAGING {0} \
      SMON_INTERFACE_TO_USE {None} \
      SMON_TEMP_AVERAGING_SAMPLES {0} \
      SMON_MEAS0 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCCAUX_103} {SUPPLY_NUM 12}} \
      SMON_MEAS1 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCCAUX_104} {SUPPLY_NUM 13}} \
      SMON_MEAS10 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_302} {SUPPLY_NUM 17}} \
      SMON_MEAS11 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_500} {SUPPLY_NUM 18}} \
      SMON_MEAS12 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_501} {SUPPLY_NUM 19}} \
      SMON_MEAS13 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_502} {SUPPLY_NUM 20}} \
      SMON_MEAS14 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 4.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {4 V unipolar}} {NAME VCCO_503} {SUPPLY_NUM 21}} \
      SMON_MEAS15 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_700} {SUPPLY_NUM 22}} \
      SMON_MEAS16 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_701} {SUPPLY_NUM 23}} \
      SMON_MEAS17 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_702} {SUPPLY_NUM 24}} \
      SMON_MEAS18 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCO_703} {SUPPLY_NUM 25}} \
      SMON_MEAS19 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_BATT} {SUPPLY_NUM 16}} \
      SMON_MEAS2 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCC_103} {SUPPLY_NUM 10}} \
      SMON_MEAS20 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PMC} {SUPPLY_NUM 2}} \
      SMON_MEAS21 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSFP} {SUPPLY_NUM 3}} \
      SMON_MEAS22 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_PSLP} {SUPPLY_NUM 4}} \
      SMON_MEAS23 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_RAM} {SUPPLY_NUM 15}} \
      SMON_MEAS24 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCC_SOC} {SUPPLY_NUM 5}} \
      SMON_MEAS3 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVCC_104} {SUPPLY_NUM 11}} \
      SMON_MEAS4 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVTT_103} {SUPPLY_NUM 8}} \
      SMON_MEAS5 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME GTYP_AVTT_104} {SUPPLY_NUM 9}} \
      SMON_MEAS6 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX} {SUPPLY_NUM 0}} \
      SMON_MEAS7 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX_PMC} {SUPPLY_NUM 1}} \
      SMON_MEAS8 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCAUX_SMON} {SUPPLY_NUM 14}} \
      SMON_MEAS9 {{ALARM_ENABLE 1} {ALARM_LOWER 0.00} {ALARM_UPPER 2.00} {AVERAGE_EN 0} {ENABLE 1} {MODE {2 V unipolar}} {NAME VCCINT} {SUPPLY_NUM 7}} \
      SMON_VAUX_CH0 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 1} {AVERAGE_EN 0} {ENABLE 1} {IO_N GND_501} {IO_P PMC_MIO22_501} {MODE {1 V unipolar}} {NAME VAUX_CH0} {SUPPLY_NUM 26}} \
      SMON_VAUX_CH1 {{ALARM_ENABLE 0} {ALARM_LOWER 0} {ALARM_UPPER 1} {AVERAGE_EN 0} {ENABLE 1} {IO_N GND_501} {IO_P PMC_MIO23_501} {MODE {1 V unipolar}} {NAME VAUX_CH1} {SUPPLY_NUM 27}} \
    } \
  ] $CIPS_0


 # Create instance: cips_noc, and set properties
  set cips_noc [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 cips_noc ]
  set_property -dict [list \
    CONFIG.NUM_CLKS {9} \
    CONFIG.NUM_MC {0} \
    CONFIG.NUM_MI {0} \
    CONFIG.NUM_NMI {9} \
    CONFIG.NUM_NSI {0} \
    CONFIG.NUM_SI {8} \
  ] $cips_noc

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M00_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M01_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M02_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M03_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M04_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M05_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M06_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M07_INI]

  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
 ] [get_bd_intf_pins /cips_noc/M08_INI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M04_INI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}} M00_INI { read_bw {128} write_bw {128}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /cips_noc/S00_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M01_INI { read_bw {128} write_bw {128}} M04_INI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /cips_noc/S01_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M02_INI { read_bw {128} write_bw {128}} M04_INI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /cips_noc/S02_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M03_INI { read_bw {128} write_bw {128}} M04_INI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_cci} \
 ] [get_bd_intf_pins /cips_noc/S03_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M05_INI { read_bw {128} write_bw {128}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /cips_noc/S04_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M06_INI { read_bw {128} write_bw {128}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_nci} \
 ] [get_bd_intf_pins /cips_noc/S05_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M07_INI { read_bw {128} write_bw {128}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_rpu} \
 ] [get_bd_intf_pins /cips_noc/S06_AXI]

  set_property -dict [ list \
   CONFIG.DATA_WIDTH {128} \
   CONFIG.CONNECTIONS {M08_INI { read_bw {128} write_bw {128}} M04_INI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}}} \
   CONFIG.DEST_IDS {} \
   CONFIG.NOC_PARAMS {} \
   CONFIG.CATEGORY {ps_pmc} \
 ] [get_bd_intf_pins /cips_noc/S07_AXI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {} \
 ] [get_bd_pins /cips_noc/aclk0]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S00_AXI} \
 ] [get_bd_pins /cips_noc/aclk1]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S01_AXI} \
 ] [get_bd_pins /cips_noc/aclk2]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S02_AXI} \
 ] [get_bd_pins /cips_noc/aclk3]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S03_AXI} \
 ] [get_bd_pins /cips_noc/aclk4]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S04_AXI} \
 ] [get_bd_pins /cips_noc/aclk5]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S05_AXI} \
 ] [get_bd_pins /cips_noc/aclk6]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S06_AXI} \
 ] [get_bd_pins /cips_noc/aclk7]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {S07_AXI} \
 ] [get_bd_pins /cips_noc/aclk8]


# Create instance: noc_ddr, and set properties
 set noc_ddr4_0 [create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 noc_ddr4_0 ]
  set_property -dict [ list \
    CONFIG.CH0_DDR4_0_BOARD_INTERFACE {ddr4_bank0} \
    CONFIG.sys_clk0_BOARD_INTERFACE {ddr4_bank0_sys_clk} \
    CONFIG.MC_CHAN_REGION0 {DDR_LOW0} \
    CONFIG.MC_CHAN_REGION1 {DDR_LOW1} \
    CONFIG.NUM_SI {0} \
    CONFIG.NUM_MI {0} \
    CONFIG.NUM_NSI {4} \
    CONFIG.NUM_CLKS {0} \
    CONFIG.NUM_MC {1} \
    CONFIG.NUM_MCP {4} \
  ] $noc_ddr4_0

set_property -dict [ list \
  CONFIG.INI_STRATEGY {auto} \
  CONFIG.CONNECTIONS {MC_0 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  ] [get_bd_intf_pins /noc_ddr4_0/S00_INI]

set_property -dict [ list \
  CONFIG.INI_STRATEGY {auto} \
  CONFIG.CONNECTIONS {MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  ] [get_bd_intf_pins /noc_ddr4_0/S01_INI]

set_property -dict [ list \
  CONFIG.INI_STRATEGY {auto} \
  CONFIG.CONNECTIONS {MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  ] [get_bd_intf_pins /noc_ddr4_0/S02_INI]

set_property -dict [ list \
  CONFIG.INI_STRATEGY {auto} \
  CONFIG.CONNECTIONS {MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  ] [get_bd_intf_pins /noc_ddr4_0/S03_INI]

# Create instance: noc_lpddr4, and set properties
# set noc_lpddr4 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 noc_lpddr4 ]
#   set_property -dict [list \
#     CONFIG.CH0_LPDDR4_0_BOARD_INTERFACE {ch0_lpddr4_trip2} \
#     CONFIG.CH0_LPDDR4_1_BOARD_INTERFACE {ch0_lpddr4_trip3} \
#     CONFIG.CH1_LPDDR4_0_BOARD_INTERFACE {ch1_lpddr4_trip2} \
#     CONFIG.CH1_LPDDR4_1_BOARD_INTERFACE {ch1_lpddr4_trip3} \
#     CONFIG.MC_CHAN_REGION0 {DDR_CH1} \
#     CONFIG.NUM_CLKS {0} \
#     CONFIG.NUM_MI {0} \
#     CONFIG.NUM_NSI {4} \
#     CONFIG.NUM_SI {0} \
#     CONFIG.NUM_MCP {4} \
#     CONFIG.sys_clk0_BOARD_INTERFACE {lpddr4_clk2} \
#     CONFIG.sys_clk1_BOARD_INTERFACE {lpddr4_clk3} \
#   ] $noc_lpddr4

  # set_property -dict [ list \
  #   CONFIG.CONNECTIONS {MC_0 { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}} } \
  #   ] [get_bd_intf_pins /noc_lpddr4/S00_INI]

  # set_property -dict [ list \
  #   CONFIG.INI_STRATEGY {auto} \
  #   CONFIG.CONNECTIONS {MC_1 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  #   ] [get_bd_intf_pins /noc_lpddr4/S01_INI]

  # set_property -dict [ list \
  #   CONFIG.INI_STRATEGY {auto} \
  #   CONFIG.CONNECTIONS {MC_2 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  #   ] [get_bd_intf_pins /noc_lpddr4/S02_INI]

  # set_property -dict [ list \
  #   CONFIG.INI_STRATEGY {auto} \
  #   CONFIG.CONNECTIONS {MC_3 { read_bw {5} write_bw {5} read_avg_burst {4} write_avg_burst {4}} } \
  #   ] [get_bd_intf_pins /noc_lpddr4/S03_INI]

  # Create instance: ai_engine_0, and set properties
  set ai_engine_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:ai_engine:2.0 ai_engine_0 ]
  set_property -dict [list \
    CONFIG.CLK_NAMES {} \
    CONFIG.FIFO_TYPE_MI_AXIS {} \
    CONFIG.FIFO_TYPE_SI_AXIS {} \
    CONFIG.NAME_MI_AXIS {} \
    CONFIG.NAME_SI_AXIS {} \
    CONFIG.NUM_CLKS {0} \
    CONFIG.NUM_MI_AXI {0} \
    CONFIG.NUM_MI_AXIS {0} \
    CONFIG.NUM_SI_AXI {1} \
    CONFIG.NUM_SI_AXIS {0} \
  ] $ai_engine_0

  set_property -dict [ list \
   CONFIG.CATEGORY {NOC} \
 ] [get_bd_intf_pins /ai_engine_0/S00_AXI]

 # Create instance: aggr_noc, and set properties
  set aggr_noc [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 aggr_noc ]
  set_property -dict [list \
    CONFIG.NUM_MI {0} \
    CONFIG.NUM_NMI {4} \
    CONFIG.NUM_NSI {8} \
    CONFIG.NUM_SI {0} \
  ] $aggr_noc


  set_property -dict [ list \
   CONFIG.INI_STRATEGY {auto} \
   CONFIG.CONNECTIONS {M00_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S00_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M01_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S01_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M02_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S02_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M03_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S03_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M00_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S04_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M00_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S05_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M00_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S06_INI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS {M00_INI { read_bw {128} write_bw {128}}} \
 ] [get_bd_intf_pins /aggr_noc/S07_INI]

  # Create instance: ConfigNoc, and set properties
  set ConfigNoc [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_noc:1.1 ConfigNoc ]
  set_property -dict [list \
    CONFIG.NUM_CLKS {1} \
    CONFIG.NUM_MI {1} \
    CONFIG.NUM_NSI {1} \
    CONFIG.NUM_SI {0} \
  ] $ConfigNoc

  set_property -dict [ list \
   CONFIG.CATEGORY {aie} \
 ] [get_bd_intf_pins /ConfigNoc/M00_AXI]

  set_property -dict [ list \
   CONFIG.CONNECTIONS { M00_AXI { read_bw {128} write_bw {128} read_avg_burst {4} write_avg_burst {4}}} \
 ] [get_bd_intf_pins /ConfigNoc/S00_INI]

  set_property -dict [ list \
   CONFIG.ASSOCIATED_BUSIF {M00_AXI} \
 ] [get_bd_pins /ConfigNoc/aclk0]

 set axi_intc_parent [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_intc axi_intc_parent ]
 set_property -dict [ list \
  CONFIG.C_IRQ_CONNECTION {1} \
  CONFIG.C_ASYNC_INTR  {0xFFFFFFFF} \
  ] $axi_intc_parent

# Create instance: clk_wizard_0, and set properties
set clk_wizard_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:clk_wizard clk_wizard_0 ]
set_property -dict [ list \
CONFIG.CLKOUT2_DIVIDE {20.000000} \
CONFIG.CLKOUT3_DIVIDE {10.000000} \
CONFIG.CLKOUT_DRIVES {BUFG,BUFG,BUFG,BUFG,BUFG,BUFG,BUFG} \
CONFIG.CLKOUT_DYN_PS {None,None,None,None,None,None,None} \
CONFIG.CLKOUT_MATCHED_ROUTING {false,false,false,false,false,false,false} \
CONFIG.CLKOUT_PORT {clk_out1,clk_out2,clk_out3,clk_out4,clk_out5,clk_out6,clk_out7} \
CONFIG.CLKOUT_REQUESTED_DUTY_CYCLE {50.000,50.000,50.000,50.000,50.000,50.000,50.000} \
CONFIG.CLKOUT_REQUESTED_OUT_FREQUENCY {104.167,156.25,312.5,78.125,208.33,416.67,625} \
CONFIG.CLKOUT_REQUESTED_PHASE {0.000,0.000,0.000,0.000,0.000,0.000,0.000} \
CONFIG.CLKOUT_USED {true,true,true,true,true,true,true} \
CONFIG.JITTER_SEL {Min_O_Jitter} \
CONFIG.RESET_TYPE {ACTIVE_LOW} \
CONFIG.USE_LOCKED {true} \
CONFIG.USE_PHASE_ALIGNMENT {true} \
CONFIG.USE_RESET {true} \
CONFIG.PRIM_IN_FREQ.VALUE_SRC USER \
CONFIG.PRIM_SOURCE {Global_buffer} \
CONFIG.PRIM_IN_FREQ {99.999908} \
] $clk_wizard_0

# Create instance: psr_104mh, and set properties
set psr_104mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_104mhz ]

# Create instance: psr_156mh, and set properties
set psr_156mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_156mhz ]

# Create instance: psr_312mh, and set properties
set psr_312mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_312mhz ]

# Create instance: psr_78mh, and set properties
set psr_78mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_78mhz ]

# Create instance: psr_208mh, and set properties
set psr_208mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_208mhz ]

# Create instance: psr_416mh, and set properties
set psr_416mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_416mhz ]

# Create instance: psr_625mh, and set properties
set psr_625mhz [ create_bd_cell -type ip -vlnv xilinx.com:ip:proc_sys_reset psr_625mhz ]

# Create instance: smartconnect_1, and set properties
set icn_ctrl_1 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect icn_ctrl_1 ]
set_property -dict [ list \
  CONFIG.NUM_CLKS {2} \
  CONFIG.NUM_MI {2} \
  CONFIG.NUM_SI {1} \
  ] $icn_ctrl_1

set icn_ctrl_2 [ create_bd_cell -type ip -vlnv xilinx.com:ip:smartconnect icn_ctrl_2 ]
set_property -dict [ list \
  CONFIG.NUM_CLKS {1} \
  CONFIG.NUM_MI {1} \
  CONFIG.NUM_SI {1} \
  ] $icn_ctrl_2

set dummy_slave_0 [ create_bd_cell -type ip -vlnv xilinx.com:ip:axi_vip dummy_slave_0 ]
set_property -dict [ list \
  CONFIG.INTERFACE_MODE {SLAVE} \
  CONFIG.PROTOCOL {AXI4LITE} \
  ] $dummy_slave_0

# Create interface connections
connect_bd_intf_net -intf_net CIPS_0_IF_PMC_NOC_AXI_0 [get_bd_intf_pins CIPS_0/PMC_NOC_AXI_0] [get_bd_intf_pins cips_noc/S07_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_0 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_0] [get_bd_intf_pins cips_noc/S00_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_1 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_1] [get_bd_intf_pins cips_noc/S01_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_2 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_2] [get_bd_intf_pins cips_noc/S02_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_CCI_3 [get_bd_intf_pins CIPS_0/FPD_CCI_NOC_3] [get_bd_intf_pins cips_noc/S03_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_0 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_0] [get_bd_intf_pins cips_noc/S04_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_NCI_1 [get_bd_intf_pins CIPS_0/FPD_AXI_NOC_1] [get_bd_intf_pins cips_noc/S05_AXI]
connect_bd_intf_net -intf_net CIPS_0_IF_PS_NOC_RPU_0 [get_bd_intf_pins CIPS_0/LPD_AXI_NOC_0] [get_bd_intf_pins cips_noc/S06_AXI]
connect_bd_intf_net -intf_net ConfigNoc_M00_AXI [get_bd_intf_pins ConfigNoc/M00_AXI] [get_bd_intf_pins ai_engine_0/S00_AXI]
connect_bd_intf_net -intf_net cips_noc_M00_INI [get_bd_intf_pins aggr_noc/S00_INI] [get_bd_intf_pins cips_noc/M00_INI]
connect_bd_intf_net -intf_net cips_noc_M01_INI [get_bd_intf_pins aggr_noc/S01_INI] [get_bd_intf_pins cips_noc/M01_INI]
connect_bd_intf_net -intf_net cips_noc_M02_INI [get_bd_intf_pins aggr_noc/S02_INI] [get_bd_intf_pins cips_noc/M02_INI]
connect_bd_intf_net -intf_net cips_noc_M03_INI [get_bd_intf_pins aggr_noc/S03_INI] [get_bd_intf_pins cips_noc/M03_INI]
connect_bd_intf_net -intf_net cips_noc_M04_INI [get_bd_intf_pins ConfigNoc/S00_INI] [get_bd_intf_pins cips_noc/M04_INI]
connect_bd_intf_net -intf_net cips_noc_M05_INI [get_bd_intf_pins aggr_noc/S04_INI] [get_bd_intf_pins cips_noc/M05_INI]
connect_bd_intf_net -intf_net cips_noc_M06_INI [get_bd_intf_pins aggr_noc/S05_INI] [get_bd_intf_pins cips_noc/M06_INI]
connect_bd_intf_net -intf_net cips_noc_M07_INI [get_bd_intf_pins aggr_noc/S06_INI] [get_bd_intf_pins cips_noc/M07_INI]
connect_bd_intf_net -intf_net cips_noc_M08_INI [get_bd_intf_pins aggr_noc/S07_INI] [get_bd_intf_pins cips_noc/M08_INI]
connect_bd_intf_net -intf_net CIPS_0_M_AXI_GP0 [get_bd_intf_pins CIPS_0/M_AXI_FPD] [get_bd_intf_pins icn_ctrl_1/S00_AXI]
connect_bd_intf_net -intf_net ddr4_bank0_sys_clk_1 [get_bd_intf_ports ddr4_bank0_sys_clk] [get_bd_intf_pins noc_ddr4_0/sys_clk0]
# connect_bd_intf_net -intf_net lpddr4_clk2_1 [get_bd_intf_ports lpddr4_clk2] [get_bd_intf_pins noc_lpddr4/sys_clk0]
# connect_bd_intf_net -intf_net lpddr4_clk3_1 [get_bd_intf_ports lpddr4_clk3] [get_bd_intf_pins noc_lpddr4/sys_clk1]
connect_bd_intf_net -intf_net icn_ctrl_1_M00_AXI [get_bd_intf_pins axi_intc_parent/s_axi] [get_bd_intf_pins icn_ctrl_1/M00_AXI]
connect_bd_intf_net -intf_net icn_ctrl_1_M01_AXI [get_bd_intf_pins icn_ctrl_1/M01_AXI] [get_bd_intf_pins icn_ctrl_2/S00_AXI]
connect_bd_intf_net -intf_net icn_ctrl_2_M00_AXI [get_bd_intf_pins icn_ctrl_2/M00_AXI] [get_bd_intf_pins dummy_slave_0/S_AXI]
connect_bd_intf_net -intf_net axi_noc_0_CH0_DDR4_0 [get_bd_intf_ports ddr4_bank0] [get_bd_intf_pins noc_ddr4_0/CH0_DDR4_0]
# connect_bd_intf_net -intf_net axi_noc_0_CH1_LPDDR4_0 [get_bd_intf_ports ch1_lpddr4_trip1] [get_bd_intf_pins noc_ddr4_0/CH1_LPDDR4_0]
# connect_bd_intf_net -intf_net noc_lpddr4_CH0_LPDDR4_0 [get_bd_intf_ports ch0_lpddr4_trip2] [get_bd_intf_pins noc_lpddr4/CH0_LPDDR4_0]
# connect_bd_intf_net -intf_net noc_lpddr4_CH0_LPDDR4_1 [get_bd_intf_ports ch0_lpddr4_trip3] [get_bd_intf_pins noc_lpddr4/CH0_LPDDR4_1]
# connect_bd_intf_net -intf_net noc_lpddr4_CH1_LPDDR4_0 [get_bd_intf_ports ch1_lpddr4_trip2] [get_bd_intf_pins noc_lpddr4/CH1_LPDDR4_0]
# connect_bd_intf_net -intf_net noc_lpddr4_CH1_LPDDR4_1 [get_bd_intf_ports ch1_lpddr4_trip3] [get_bd_intf_pins noc_lpddr4/CH1_LPDDR4_1]
connect_bd_intf_net -intf_net axi_noc_0_M00_INI [get_bd_intf_pins aggr_noc/M00_INI] [get_bd_intf_pins noc_ddr4_0/S00_INI]
connect_bd_intf_net -intf_net axi_noc_0_M01_INI [get_bd_intf_pins aggr_noc/M01_INI] [get_bd_intf_pins noc_ddr4_0/S01_INI]
connect_bd_intf_net -intf_net axi_noc_0_M02_INI [get_bd_intf_pins aggr_noc/M02_INI] [get_bd_intf_pins noc_ddr4_0/S02_INI]
connect_bd_intf_net -intf_net axi_noc_0_M03_INI [get_bd_intf_pins aggr_noc/M03_INI] [get_bd_intf_pins noc_ddr4_0/S03_INI]
# connect_bd_intf_net -intf_net axi_noc_0_M04_INI [get_bd_intf_pins aggr_noc/M04_INI] [get_bd_intf_pins noc_lpddr4/S00_INI]
# connect_bd_intf_net -intf_net axi_noc_0_M05_INI [get_bd_intf_pins aggr_noc/M05_INI] [get_bd_intf_pins noc_lpddr4/S01_INI]
# connect_bd_intf_net -intf_net axi_noc_0_M06_INI [get_bd_intf_pins aggr_noc/M06_INI] [get_bd_intf_pins noc_lpddr4/S02_INI]
# connect_bd_intf_net -intf_net axi_noc_0_M07_INI [get_bd_intf_pins aggr_noc/M07_INI] [get_bd_intf_pins noc_lpddr4/S03_INI]


# Create port connections
connect_bd_net -net CIPS_0_pl_clk0 [get_bd_pins CIPS_0/pl0_ref_clk] [get_bd_pins clk_wizard_0/clk_in1]
connect_bd_net -net CIPS_0_pl_resetn1 [get_bd_pins CIPS_0/pl0_resetn] [get_bd_pins clk_wizard_0/resetn] [get_bd_pins psr_156mhz/ext_reset_in] [get_bd_pins psr_104mhz/ext_reset_in] [get_bd_pins psr_208mhz/ext_reset_in] [get_bd_pins psr_312mhz/ext_reset_in] [get_bd_pins psr_416mhz/ext_reset_in] [get_bd_pins psr_625mhz/ext_reset_in] [get_bd_pins psr_78mhz/ext_reset_in]
connect_bd_net -net CIPS_0_ps_pmc_noc_axi0_clk [get_bd_pins CIPS_0/pmc_axi_noc_axi0_clk] [get_bd_pins cips_noc/aclk8]
connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi0_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi0_clk] [get_bd_pins cips_noc/aclk1]
connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi1_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi1_clk] [get_bd_pins cips_noc/aclk2]
connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi2_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi2_clk] [get_bd_pins cips_noc/aclk3]
connect_bd_net -net CIPS_0_ps_ps_noc_cci_axi3_clk [get_bd_pins CIPS_0/fpd_cci_noc_axi3_clk] [get_bd_pins cips_noc/aclk4]
connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi0_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi0_clk] [get_bd_pins cips_noc/aclk5]
connect_bd_net -net CIPS_0_ps_ps_noc_nci_axi1_clk [get_bd_pins CIPS_0/fpd_axi_noc_axi1_clk] [get_bd_pins cips_noc/aclk6]
connect_bd_net -net CIPS_0_ps_ps_noc_rpu_axi0_clk [get_bd_pins CIPS_0/lpd_axi_noc_clk] [get_bd_pins cips_noc/aclk7]
connect_bd_net -net ai_engine_0_s00_axi_aclk [get_bd_pins ConfigNoc/aclk0] [get_bd_pins ai_engine_0/s00_axi_aclk]
connect_bd_net -net axi_intc_3_irq [get_bd_pins CIPS_0/pl_ps_irq0] [get_bd_pins axi_intc_parent/irq]
connect_bd_net -net clk_wizard_0_clk_out1 [get_bd_pins clk_wizard_0/clk_out1] [get_bd_pins ConfigNoc/aclk5] [get_bd_pins CIPS_0/m_axi_fpd_aclk] [get_bd_pins axi_intc_parent/s_axi_aclk] [get_bd_pins cips_noc/aclk0]  [get_bd_pins icn_ctrl_1/aclk] [get_bd_pins icn_ctrl_1/aclk1] [get_bd_pins icn_ctrl_2/aclk] [get_bd_pins psr_104mhz/slowest_sync_clk] [get_bd_pins dummy_slave_0/aclk] 
connect_bd_net -net clk_wizard_0_clk_out2 [get_bd_pins clk_wizard_0/clk_out2] [get_bd_pins psr_156mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_clk_out3 [get_bd_pins clk_wizard_0/clk_out3] [get_bd_pins psr_312mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_clk_out4 [get_bd_pins clk_wizard_0/clk_out4] [get_bd_pins psr_78mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_clk_out5 [get_bd_pins clk_wizard_0/clk_out5] [get_bd_pins psr_208mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_clk_out6 [get_bd_pins clk_wizard_0/clk_out6] [get_bd_pins psr_416mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_clk_out7 [get_bd_pins clk_wizard_0/clk_out7] [get_bd_pins psr_625mhz/slowest_sync_clk]
connect_bd_net -net clk_wizard_0_locked [get_bd_pins clk_wizard_0/locked] [get_bd_pins psr_104mhz/dcm_locked] [get_bd_pins psr_156mhz/dcm_locked] [get_bd_pins psr_208mhz/dcm_locked] [get_bd_pins psr_312mhz/dcm_locked] [get_bd_pins psr_416mhz/dcm_locked] [get_bd_pins psr_600mhz/dcm_locked] [get_bd_pins psr_78mhz/dcm_locked]
connect_bd_net -net psr_104mhz_peripheral_aresetn [get_bd_pins axi_intc_parent/s_axi_aresetn] [get_bd_pins icn_ctrl_1/aresetn] [get_bd_pins icn_ctrl_2/aresetn] [get_bd_pins psr_104mhz/peripheral_aresetn] [get_bd_pins dummy_slave_0/aresetn] 


# Create address segments
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_AXI_NOC_1] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW1] -force
  # assign_bd_address -offset 0x050000000000 -range 0x000200000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_0] [get_bd_addr_segs noc_lpddr4/S00_INI/C0_DDR_CH1x2] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs noc_ddr4_0/S01_INI/C1_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs noc_ddr4_0/S01_INI/C1_DDR_LOW1] -force
  # assign_bd_address -offset 0x050000000000 -range 0x000200000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_1] [get_bd_addr_segs noc_lpddr4/S01_INI/C1_DDR_CH1x2] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs noc_ddr4_0/S02_INI/C2_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs noc_ddr4_0/S02_INI/C2_DDR_LOW1] -force
  # assign_bd_address -offset 0x050000000000 -range 0x000200000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_2] [get_bd_addr_segs noc_lpddr4/S02_INI/C2_DDR_CH1x2] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs noc_ddr4_0/S03_INI/C3_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs noc_ddr4_0/S03_INI/C3_DDR_LOW1] -force
  # assign_bd_address -offset 0x050000000000 -range 0x000200000000 -target_address_space [get_bd_addr_spaces CIPS_0/FPD_CCI_NOC_3] [get_bd_addr_segs noc_lpddr4/S03_INI/C3_DDR_CH1x2] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/LPD_AXI_NOC_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW1] -force
  assign_bd_address -offset 0xA5000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/M_AXI_FPD] [get_bd_addr_segs axi_intc_parent/S_AXI/Reg] -force
  assign_bd_address -offset 0xA4000000 -range 0x00010000 -target_address_space [get_bd_addr_spaces CIPS_0/M_AXI_FPD] [get_bd_addr_segs dummy_slave_0/S_AXI/Reg] -force
  assign_bd_address -offset 0x020000000000 -range 0x000100000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs ai_engine_0/S00_AXI/AIE_ARRAY_0] -force
  assign_bd_address -offset 0x00000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW0] -force
  assign_bd_address -offset 0x000800000000 -range 0x80000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs noc_ddr4_0/S00_INI/C0_DDR_LOW1] -force
  # assign_bd_address -offset 0x050000000000 -range 0x000200000000 -target_address_space [get_bd_addr_spaces CIPS_0/PMC_NOC_AXI_0] [get_bd_addr_segs noc_lpddr4/S00_INI/C0_DDR_CH1x2] -force
  assign_bd_address
# Restore current instance
current_bd_instance $oldCurInst
}
create_root_design ""
regenerate_bd_layout
save_bd_design

