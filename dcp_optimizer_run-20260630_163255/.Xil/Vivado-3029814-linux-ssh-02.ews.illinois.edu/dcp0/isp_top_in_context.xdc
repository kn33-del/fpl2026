create_clock -period 1.570 [get_ports {aclk}]
set_property -quiet IS_IP_OOC_CELL TRUE [get_cells -of [get_ports -no_traverse -quiet aclk]]
