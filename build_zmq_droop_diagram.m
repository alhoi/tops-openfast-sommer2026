%% build_zmq_droop_diagram.m
% Builds an illustrative Simulink block diagram of the coupling used in the
% OpenFAST/LEOGO co-simulation:
%
%   LEOGO grid (COI frequency)  ->  low-pass filter (2nd order)
%     ->  droop  +  virtual inertia  ->  saturation  ->  support envelope
%     ->  ZMQ TorqueOffset  ->  ROSCO  ->  (back into the grid).
%
% This is a DIAGRAM for figures, not a runnable plant model: the grid/turbine
% and the ZMQ link are placeholder Subsystem blocks that close the loop so the
% signal flow reads correctly. The control-path blocks (LPF, droop, inertia,
% saturation) carry the real parameter values from
% casestudies/dyn_sim/rosco_zmq_grid_coupling.py.
%
% Run:     build_zmq_droop_diagram
% Output:  zmq_droop_coupling.slx  (opened and saved next to this script)
% Export:  print(['-s' mdl],'-dpng','-r300',[mdl '.png'])   % after running

mdl = 'zmq_droop_coupling';

% --- Parameters (edit here; they appear as labels on the blocks) ----------
fc            = 0.5;             % frequency-measurement LPF corner [Hz]
tau           = 1/(2*pi*fc);    % first-order time constant [s]
f_nom         = 50;             % nominal grid frequency [Hz]
K_droop       = 2e7;            % droop gain [Nm/Hz]
K_inertia     = 5e6;            % virtual-inertia gain [Nm.s/Hz]
Tmax          = 3e6;            % torque-offset saturation [Nm]
support_start = 10;             % support ramp-in time [s]

% Expose to the base workspace so the symbolic block labels resolve cleanly.
assignin('base','tau',tau);
assignin('base','f_nom',f_nom);
assignin('base','K_droop',K_droop);
assignin('base','K_inertia',K_inertia);
assignin('base','Tmax',Tmax);
assignin('base','support_start',support_start);

% --- Fresh model ----------------------------------------------------------
if ~isempty(find_system('SearchDepth',0,'Name',mdl))
    close_system(mdl,0);
end
new_system(mdl);
open_system(mdl);
load_system('simulink');   % ensure the library block paths resolve

add = @(src,name,pos,varargin) add_block(src,[mdl '/' name], ...
        'Position',pos,varargin{:});

% --- Blocks (rough positions; arrangeSystem tidies them up at the end) ----
add('simulink/Ports & Subsystems/Subsystem', ...
    'LEOGO + UIC + OpenFAST-ROSCO',[40 300 240 390]);
add('simulink/Continuous/Transfer Fcn','LPF1',[320 80 410 130], ...
    'Numerator','[1]','Denominator','[tau 1]');
add('simulink/Continuous/Transfer Fcn','LPF2',[450 80 540 130], ...
    'Numerator','[1]','Denominator','[tau 1]');
add('simulink/Sources/Constant','f_nom',[450 190 540 220],'Value','f_nom');
add('simulink/Math Operations/Sum','delta f',[590 88 620 118],'Inputs','+-');
add('simulink/Math Operations/Gain','droop gain',[660 80 730 120], ...
    'Gain','-K_droop');
add('simulink/Continuous/Derivative','derivative',[660 180 730 220]);
add('simulink/Math Operations/Gain','inertia gain',[770 180 850 220], ...
    'Gain','-K_inertia');
add('simulink/Math Operations/Sum','sum torque',[890 108 920 138], ...
    'Inputs','++');
add('simulink/Discontinuities/Saturation','saturation',[960 100 1030 140], ...
    'UpperLimit','Tmax','LowerLimit','-Tmax');
add('simulink/Sources/Step','support enable',[960 10 1020 50], ...
    'Time','support_start','Before','0','After','1');
add('simulink/Math Operations/Product','envelope',[1070 100 1110 140]);
add('simulink/Ports & Subsystems/Subsystem', ...
    'ZMQ to ROSCO',[1160 90 1360 160]);

% --- Wires ----------------------------------------------------------------
ar = {'autorouting','on'};
l = add_line(mdl,'LEOGO + UIC + OpenFAST-ROSCO/1','LPF1/1',ar{:});
set_param(l,'Name','f_grid');
add_line(mdl,'LPF1/1','LPF2/1',ar{:});
l = add_line(mdl,'LPF2/1','delta f/1',ar{:});
set_param(l,'Name','f_filt');
add_line(mdl,'f_nom/1','delta f/2',ar{:});
add_line(mdl,'LPF2/1','derivative/1',ar{:});               % branch off f_filt
l = add_line(mdl,'delta f/1','droop gain/1',ar{:});
set_param(l,'Name','delta_f');
add_line(mdl,'derivative/1','inertia gain/1',ar{:});
add_line(mdl,'droop gain/1','sum torque/1',ar{:});
add_line(mdl,'inertia gain/1','sum torque/2',ar{:});
add_line(mdl,'sum torque/1','saturation/1',ar{:});
add_line(mdl,'saturation/1','envelope/1',ar{:});
add_line(mdl,'support enable/1','envelope/2',ar{:});
l = add_line(mdl,'envelope/1','ZMQ to ROSCO/1',ar{:});
set_param(l,'Name','delta_T');
l = add_line(mdl,'ZMQ to ROSCO/1','LEOGO + UIC + OpenFAST-ROSCO/1',ar{:});
set_param(l,'Name','delta_T_to_ROSCO');

% --- Tidy up + save -------------------------------------------------------
Simulink.BlockDiagram.arrangeSystem(mdl);
set_param(mdl,'ZoomFactor','FitSystem');
save_system(mdl);
fprintf('Saved %s.slx  (run print(''-s%s'',''-dpng'',''-r300'',''%s.png'') to export)\n', ...
        mdl, mdl, mdl);
