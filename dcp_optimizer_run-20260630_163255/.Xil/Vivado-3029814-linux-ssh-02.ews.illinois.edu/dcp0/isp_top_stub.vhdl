-- Copyright 1986-2022 Xilinx, Inc. All Rights Reserved.
-- Copyright 2022-2025 Advanced Micro Devices, Inc. All Rights Reserved.
library IEEE;
use IEEE.STD_LOGIC_1164.ALL;

entity isp_top is
  Port ( 
    s_axi_wready : out STD_LOGIC;
    m_axis_tvalid : out STD_LOGIC;
    s_axis_tuser : in STD_LOGIC_VECTOR ( 0 to 0 );
    s_axi_araddr : in STD_LOGIC_VECTOR ( 4 downto 0 );
    s_axi_awaddr : in STD_LOGIC_VECTOR ( 4 downto 0 );
    aclk : in STD_LOGIC;
    s_axi_bresp : out STD_LOGIC_VECTOR ( 1 downto 0 );
    s_axi_rdata : out STD_LOGIC_VECTOR ( 31 downto 0 );
    s_axi_bready : in STD_LOGIC;
    s_axi_wstrb : in STD_LOGIC_VECTOR ( 3 downto 0 );
    s_axi_rready : in STD_LOGIC;
    m_axis_tlast : out STD_LOGIC;
    s_axi_arvalid : in STD_LOGIC;
    m_axis_tdata : out STD_LOGIC_VECTOR ( 95 downto 0 );
    m_axis_tready : in STD_LOGIC;
    s_axi_arready : out STD_LOGIC;
    s_axi_wvalid : in STD_LOGIC;
    s_axis_tdata : in STD_LOGIC_VECTOR ( 47 downto 0 );
    aresetn : in STD_LOGIC;
    s_axis_tready : out STD_LOGIC;
    m_axis_tuser : out STD_LOGIC_VECTOR ( 0 to 0 );
    s_axi_wdata : in STD_LOGIC_VECTOR ( 31 downto 0 );
    s_axi_rresp : out STD_LOGIC_VECTOR ( 1 downto 0 );
    s_axi_rvalid : out STD_LOGIC;
    s_axi_awready : out STD_LOGIC;
    s_axi_bvalid : out STD_LOGIC;
    s_axis_tlast : in STD_LOGIC;
    s_axi_awvalid : in STD_LOGIC;
    s_axis_tvalid : in STD_LOGIC
  );

  attribute CFA_ORIENTATION : integer;
  attribute CFA_ORIENTATION of isp_top : entity is 0;
  attribute COMPONENT_BIT_WIDTH : integer;
  attribute COMPONENT_BIT_WIDTH of isp_top : entity is 8;
  attribute DECOMPANDING_FLUT_FILE : string;
  attribute DECOMPANDING_FLUT_FILE of isp_top : entity is "decompanding_flut_12_bit.mem";
  attribute DECOMPANDING_NUM_KNEE_POINTS : integer;
  attribute DECOMPANDING_NUM_KNEE_POINTS of isp_top : entity is 16;
  attribute DECOMPANDING_XLUT_FILE : string;
  attribute DECOMPANDING_XLUT_FILE of isp_top : entity is "decompanding_xlut.mem";
  attribute DECOMPANDING_YLUT_FILE : string;
  attribute DECOMPANDING_YLUT_FILE of isp_top : entity is "decompanding_ylut_12_bit.mem";
  attribute ECO_CHECKSUM : string;
  attribute ECO_CHECKSUM of isp_top : entity is "10f0d753";
  attribute GAMMA_LUT_FILE : string;
  attribute GAMMA_LUT_FILE of isp_top : entity is "lut.mem";
  attribute MAX_RESOLUTION : integer;
  attribute MAX_RESOLUTION of isp_top : entity is 4096;
  attribute M_AXIS_DATA_WIDTH : integer;
  attribute M_AXIS_DATA_WIDTH of isp_top : entity is 96;
  attribute PIXEL_BIT_WIDTH : integer;
  attribute PIXEL_BIT_WIDTH of isp_top : entity is 12;
  attribute PIXEL_PER_CYCLE : integer;
  attribute PIXEL_PER_CYCLE of isp_top : entity is 4;
  attribute S_AXIS_DATA_WIDTH : integer;
  attribute S_AXIS_DATA_WIDTH of isp_top : entity is 48;
  attribute S_AXI_ADDR_WIDTH : integer;
  attribute S_AXI_ADDR_WIDTH of isp_top : entity is 5;
  attribute S_AXI_BRESP_WIDTH : integer;
  attribute S_AXI_BRESP_WIDTH of isp_top : entity is 2;
  attribute S_AXI_DATA_WIDTH : integer;
  attribute S_AXI_DATA_WIDTH of isp_top : entity is 32;
  attribute S_AXI_RRESP_WIDTH : integer;
  attribute S_AXI_RRESP_WIDTH of isp_top : entity is 2;
  attribute S_AXI_WSTRB_WIDTH : integer;
  attribute S_AXI_WSTRB_WIDTH of isp_top : entity is 4;
  attribute TUSER_WIDTH : integer;
  attribute TUSER_WIDTH of isp_top : entity is 1;
end isp_top;

architecture stub of isp_top is
  attribute syn_black_box : boolean;
  attribute black_box_pad_pin : string;
  attribute syn_black_box of stub : architecture is true;
begin
end;
