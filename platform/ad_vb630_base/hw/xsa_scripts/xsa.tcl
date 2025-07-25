#******************************************************************************
# Copyright (C) 2020-2022 Xilinx, Inc. All rights reserved.
# Copyright (C) 2022-2024 Advanced Micro Devices, Inc. All rights reserved.
# SPDX-License-Identifier: MIT
#******************************************************************************

file mkdir build 
cd build
source ../xsa_scripts/project.tcl
source ../xsa_scripts/dr.bd.tcl
source ../xsa_scripts/pfm_decls.tcl
#For Questa Simulator
source ../data/questa_sim.tcl 

#Generating Wrapper
make_wrapper -files [get_files ./my_project/my_project.srcs/sources_1/bd/vitis_design/vitis_design.bd] -top
add_files -norecurse ./my_project/my_project.srcs/sources_1/bd/vitis_design/hdl/vitis_design_wrapper.v

#Generating Target
generate_target all [get_files ./my_project/my_project.srcs/sources_1/bd/vitis_design/vitis_design.bd]

update_compile_order -fileset sources_1
update_compile_order -fileset sim_1
set_property top vitis_design_wrapper [current_fileset] 

# Ensure that your top of synthesis module is also set as top for simulation
set_property top vitis_design_wrapper [get_filesets sim_1]

# Generate simulation top for your entire design which would include
# aggregated NOC in the form of xlnoc.bd
generate_switch_network_for_noc
update_compile_order -fileset sim_1

# Set the auto-generated <rtl_top>_sim_wrapper as the sim top
set_property top vitis_design_wrapper_sim_wrapper [get_filesets sim_1]
import_files -fileset sim_1 -norecurse ./my_project/my_project.srcs/sources_1/common/hdl/vitis_design_wrapper_sim_wrapper.v 
update_compile_order -fileset sim_1

#Generate the final simulation script which will compile
# the <syn_top>_sim_wrapper and xlnoc.bd modules also
launch_simulation -scripts_only
launch_simulation -step compile
launch_simulation -step elaborate

#Generating Emulation XSA
file mkdir hw_emu
write_hw_platform -hw_emu -file hw_emu/hw_emu.xsa

set pre_synth ""
if { $argc > 1} {
  set pre_synth [lindex $argv 2]
}

#Pre_synth Platform Flow
if {$pre_synth} {
  set_property platform.platform_state "pre_synth" [current_project]
  write_hw_platform -hw -force -file hw.xsa
} else {

  #Post_implememtation Platform
  # Synthesis Run
  launch_runs synth_1 -jobs 20
  wait_on_run synth_1

  #Implementation Run
  launch_runs impl_1 -to_step write_device_image
  wait_on_run impl_1

  open_run impl_1

  # Generating XSA
  write_hw_platform -hw -force -include_bit -file hw.xsa
}

#generate README.hw
set board admvb630

set fd [open README.hw w] 

set board [lindex $argv 0]

puts $fd "##########################################################################"
puts $fd "This is a brief document containing design specific details for : ${board}"
puts $fd "This is auto-generated by Petalinux ref-design builder created @ [clock format [clock seconds] -format {%a %b %d %H:%M:%S %Z %Y}]"
puts $fd "##########################################################################"

set board_part [get_board_parts [current_board_part -quiet]]
if { $board_part != ""} {
  puts $fd "BOARD: $board_part" 
}

set design_name [get_property NAME [get_bd_designs]]
puts $fd "BLOCK DESIGN: $design_name" 

set columns {%40s%30s%15s%50s}
puts $fd [string repeat - 150]
puts $fd [format $columns "MODULE INSTANCE NAME" "IP TYPE" "IP VERSION" "IP"]
puts $fd [string repeat - 150]
foreach ip [get_ips] {
  set catlg_ip [get_ipdefs -all [get_property IPDEF $ip]] 
  puts $fd [format $columns [get_property NAME $ip] [get_property NAME $catlg_ip] [get_property VERSION $catlg_ip] [get_property VLNV $catlg_ip]]
}
close $fd

cd ..
