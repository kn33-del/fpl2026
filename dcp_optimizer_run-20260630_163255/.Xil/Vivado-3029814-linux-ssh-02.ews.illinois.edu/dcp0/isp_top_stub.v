// Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
// Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.

// This empty module with port declaration file causes synthesis tools to infer a black box for IP.
// Please paste the declaration into a Verilog source file or add the file as an additional source.
(* CFA_ORIENTATION = "0" *) (* COMPONENT_BIT_WIDTH = "8" *) (* DECOMPANDING_FLUT_FILE = "decompanding_flut_12_bit.mem" *) 
(* DECOMPANDING_NUM_KNEE_POINTS = "16" *) (* DECOMPANDING_XLUT_FILE = "decompanding_xlut.mem" *) (* DECOMPANDING_YLUT_FILE = "decompanding_ylut_12_bit.mem" *) 
(* ECO_CHECKSUM = "10f0d753" *) (* GAMMA_LUT_FILE = "lut.mem" *) (* MAX_RESOLUTION = "4096" *) 
(* M_AXIS_DATA_WIDTH = "96" *) (* PIXEL_BIT_WIDTH = "12" *) (* PIXEL_PER_CYCLE = "4" *) 
(* S_AXIS_DATA_WIDTH = "48" *) (* S_AXI_ADDR_WIDTH = "5" *) (* S_AXI_BRESP_WIDTH = "2" *) 
(* S_AXI_DATA_WIDTH = "32" *) (* S_AXI_RRESP_WIDTH = "2" *) (* S_AXI_WSTRB_WIDTH = "4" *) 
(* TUSER_WIDTH = "1" *) 
module isp_top(s_axi_wready, m_axis_tvalid, s_axis_tuser, 
  s_axi_araddr, s_axi_awaddr, aclk, s_axi_bresp, s_axi_rdata, s_axi_bready, s_axi_wstrb, 
  s_axi_rready, m_axis_tlast, s_axi_arvalid, m_axis_tdata, m_axis_tready, s_axi_arready, 
  s_axi_wvalid, s_axis_tdata, aresetn, s_axis_tready, m_axis_tuser, s_axi_wdata, s_axi_rresp, 
  s_axi_rvalid, s_axi_awready, s_axi_bvalid, s_axis_tlast, s_axi_awvalid, s_axis_tvalid);
  output s_axi_wready;
  output m_axis_tvalid;
  input [0:0]s_axis_tuser;
  input [4:0]s_axi_araddr;
  input [4:0]s_axi_awaddr;
  input aclk /* synthesis syn_isclock = 1 */;
  output [1:0]s_axi_bresp;
  output [31:0]s_axi_rdata;
  input s_axi_bready;
  input [3:0]s_axi_wstrb;
  input s_axi_rready;
  output m_axis_tlast;
  input s_axi_arvalid;
  output [95:0]m_axis_tdata;
  input m_axis_tready;
  output s_axi_arready;
  input s_axi_wvalid;
  input [47:0]s_axis_tdata;
  input aresetn;
  output s_axis_tready;
  output [0:0]m_axis_tuser;
  input [31:0]s_axi_wdata;
  output [1:0]s_axi_rresp;
  output s_axi_rvalid;
  output s_axi_awready;
  output s_axi_bvalid;
  input s_axis_tlast;
  input s_axi_awvalid;
  input s_axis_tvalid;
endmodule
