set_property SRC_FILE_INFO {cfile:/tmp/1 rfile:../../../../../tmp/1 id:1} [current_design]
set_property src_info {type:XDC file:1 line:1 export:INPUT save:INPUT read:READ} [current_design]
# Mini-ISP FPL'26 Contest Benchmark Constraints
set_property src_info {type:XDC file:1 line:2 export:INPUT save:INPUT read:READ} [current_design]
# Target: xcvu3p-ffvc1517-2-e
set_property src_info {type:XDC file:1 line:3 export:INPUT save:INPUT read:READ} [current_design]
# Out-of-context (no IO assignments)
set_property src_info {type:XDC file:1 line:4 export:INPUT save:INPUT read:READ} [current_design]

set_property src_info {type:XDC file:1 line:5 export:INPUT save:INPUT read:READ} [current_design]
# Primary clock: ~637 MHz = 1.570 ns period
set_property src_info {type:XDC file:1 line:6 export:INPUT save:INPUT read:READ} [current_design]
# (Stays within BRAM min period of 1.569ns on US+)
set_property src_info {type:XDC file:1 line:7 export:INPUT save:INPUT read:READ} [current_design]
create_clock -period 1.570 -name clk_fpl26contest [get_ports aclk]
set_property src_info {type:XDC file:1 line:8 export:INPUT save:INPUT read:READ} [current_design]

set_property src_info {type:XDC file:1 line:8 export:INPUT save:INPUT read:READ} [current_design]

set_property src_info {type:XDC file:1 line:8 export:INPUT save:INPUT read:READ} [current_design]

set_property src_info {type:XDC file:1 line:8 export:INPUT save:INPUT read:READ} [current_design]

