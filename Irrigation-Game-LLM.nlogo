extensions [py]

turtles-own [
  pos-label           ; "A" "B" "C" "D" "E"
  contribution        ; investment this tick (0-10)
  collected           ; extraction this tick
  income              ; (10 - contribution) + collected
]

globals [
  invest              ; total contribution this tick
  pg                  ; public good (water) this tick
  pga                 ; remaining water during sequential extraction
  run-id
  institution-score
]

; -----------------------------------------------------------------------
; Setup
; -----------------------------------------------------------------------

to setup
  ca
  set institution-score -1

  py:setup py:python3
  py:run "import sys, os; sys.path.insert(0, os.getcwd())"
  py:run "import irrigation_llm_bridge as bridge"

  create-turtles 5 [
    let labels ["A" "B" "C" "D" "E"]
    set pos-label item who labels
    set contribution 0
    set collected 0
    set income 0
    set xcor -4 + (who * 2)
    set ycor 0
    set label pos-label
    set label-color white
    set size 2
    set shape "person"
    set color item who [blue green red orange violet]
  ]

  setup-python-bridge
  reset-ticks
end

to setup-python-bridge
  let now-str (word ticks)  ; simple placeholder; NetLogo doesn't have datetime natively
  let d date-and-time       ; "HH:MM:SS MM/DD/YYYY" format
  ; Build run_id from date-and-time
  let date-part substring d 9 19      ; MM/DD/YYYY
  let time-part substring d 0 8      ; HH:MM:SS
  let yy substring date-part 6 10
  let mm substring date-part 0 2
  let dd substring date-part 3 5
  let hh substring time-part 0 2
  let mi substring time-part 3 5
  let ss substring time-part 6 8
  set run-id (word yy mm dd "_" hh mi ss "_irrigation-llm")

  py:set "run_id_val"    run-id
  py:set "log_dir_val"   "logs"
  py:set "mem_len"       memory-length
  py:set "max_tok"       256
  py:set "inst_int"      institution-check-interval
  py:set "backends_val"  (list agent0-backend agent1-backend agent2-backend agent3-backend agent4-backend)
  py:set "models_val"    (list agent0-model agent1-model agent2-model agent3-model agent4-model)
  py:set "ollama_url"    ollama-base-url
  py:set "sys_ov"        system-prompt-override

  py:run "bridge.configure(run_id_val, log_dir_val, mem_len, max_tok, inst_int, backends_val, models_val, ollama_url, sys_ov)"

  ; Log experiment parameters
  py:set "p_keys" ["num_rounds" "memory_length" "institution_check_interval"
                   "agent0_backend" "agent0_model"
                   "agent1_backend" "agent1_model"
                   "agent2_backend" "agent2_model"
                   "agent3_backend" "agent3_model"
                   "agent4_backend" "agent4_model"]
  py:set "p_vals" (list num-rounds memory-length institution-check-interval
                   agent0-backend agent0-model
                   agent1-backend agent1-model
                   agent2-backend agent2-model
                   agent3-backend agent3-model
                   agent4-backend agent4-model)
  py:run "bridge.log_params(dict(zip(p_keys, p_vals)))"
end

; -----------------------------------------------------------------------
; Go
; -----------------------------------------------------------------------

to go
  if ticks >= num-rounds [ stop ]

  let cur-tick ticks
  let sys-ov system-prompt-override

  ; Phase 1 — simultaneous contribution decision
  py:set "cur_tick" cur-tick
  py:set "sys_ov_val" sys-ov

  ask turtles [
    py:set "agent_id" who
    set contribution py:runresult "bridge.decide_contribution(agent_id, cur_tick, sys_ov_val)"
    set contribution max (list 0 (min (list 10 contribution)))  ; clamp
  ]

  ; Compute total investment and public good
  set invest sum [contribution] of turtles
  set pg calpg invest
  set pga pg

  py:set "total_inv" invest
  py:set "pg_now" pg

  ; Phase 2 — sequential extraction (A → E)
  let already-taken 0
  let pos 0
  while [pos < 5] [
    ask turtle pos [
      py:set "agent_id"   who
      py:set "contrib_v"  contribution
      py:set "avail_v"    pga
      py:set "taken_v"    already-taken
      set collected py:runresult "bridge.decide_extraction(agent_id, cur_tick, contrib_v, total_inv, pg_now, avail_v, taken_v, sys_ov_val)"
      set collected max (list 0 (min (list (floor pga) collected)))  ; safety clamp
      set income (10 - contribution) + collected
    ]
    let taken-now [collected] of turtle pos
    set pga pga - taken-now
    set already-taken already-taken + taken-now
    set pos pos + 1
  ]

  ; Institution score (LLM secondary pass every N ticks)
  let c-list map [t -> [contribution] of t] sort turtles
  let x-list map [t -> [collected] of t] sort turtles
  py:set "c_list_v" c-list
  py:set "x_list_v" x-list
  let sc py:runresult "bridge.score_institution(cur_tick, c_list_v, x_list_v, sys_ov_val)"
  if sc >= 0 [ set institution-score sc ]

  tick
  update-plots
end

; -----------------------------------------------------------------------
; Helpers
; -----------------------------------------------------------------------

to-report calpg [inv]
  if inv < 10 [ report 0  ]
  if inv < 15 [ report 5  ]
  if inv < 20 [ report 20 ]
  if inv < 25 [ report 40 ]
  if inv < 30 [ report 60 ]
  if inv < 35 [ report 75 ]
  if inv < 40 [ report 85 ]
  if inv < 45 [ report 95 ]
  report 100
end

to-report mean-contribution
  if count turtles = 0 [ report 0 ]
  report mean [contribution] of turtles
end

to-report mean-extraction
  if count turtles = 0 [ report 0 ]
  report mean [collected] of turtles
end

to-report mean-income
  if count turtles = 0 [ report 0 ]
  report mean [income] of turtles
end

to-report gini-contribution
  if count turtles = 0 [ report 0 ]
  let vals [contribution] of turtles
  if mean vals = 0 [ report 0 ]
  let n length vals
  let s 0
  foreach vals [ v1 -> foreach vals [ v2 -> set s s + abs(v1 - v2) ] ]
  report s / (2 * (mean vals) * n ^ 2)
end

to-report gini-extraction
  if count turtles = 0 [ report 0 ]
  let vals [collected] of turtles
  if mean vals = 0 [ report 0 ]
  let n length vals
  let s 0
  foreach vals [ v1 -> foreach vals [ v2 -> set s s + abs(v1 - v2) ] ]
  report s / (2 * (mean vals) * n ^ 2)
end
@#$#@#$#@
GRAPHICS-WINDOW
471
10
871
411
-1
-1
39.2
1
10
1
1
1
0
0
0
1
-4
4
-4
4
0
0
1
ticks
30.0

BUTTON
8
10
230
43
NIL
setup
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
8
47
230
80
NIL
go
T
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

BUTTON
8
83
230
116
step
go
NIL
1
T
OBSERVER
NIL
NIL
NIL
NIL
1

TEXTBOX
8
122
230
140
Simulation parameters:
11
0.0
1

SLIDER
8
140
230
173
num-rounds
num-rounds
1
50
10
1
1
rounds
HORIZONTAL

SLIDER
8
176
230
209
memory-length
memory-length
1
10
5
1
1
rounds
HORIZONTAL

SLIDER
8
212
230
245
institution-check-interval
institution-check-interval
1
20
5
1
1
ticks
HORIZONTAL

TEXTBOX
8
254
230
272
Agent 0 (A — most upstream):
11
0.0
1

CHOOSER
8
272
148
317
agent0-backend
agent0-backend
"anthropic" "openai" "ollama" "google"
0

INPUTBOX
150
272
230
317
agent0-model
claude-sonnet-4-6
1
0
String

TEXTBOX
8
322
230
340
Agent 1 (B):
11
0.0
1

CHOOSER
8
340
148
385
agent1-backend
agent1-backend
"anthropic" "openai" "ollama" "google"
0

INPUTBOX
150
340
230
385
agent1-model
claude-sonnet-4-6
1
0
String

TEXTBOX
8
390
230
408
Agent 2 (C — middle):
11
0.0
1

CHOOSER
8
408
148
453
agent2-backend
agent2-backend
"anthropic" "openai" "ollama" "google"
0

INPUTBOX
150
408
230
453
agent2-model
claude-sonnet-4-6
1
0
String

TEXTBOX
8
458
230
476
Agent 3 (D):
11
0.0
1

CHOOSER
8
476
148
521
agent3-backend
agent3-backend
"anthropic" "openai" "ollama" "google"
0

INPUTBOX
150
476
230
521
agent3-model
claude-sonnet-4-6
1
0
String

TEXTBOX
8
526
230
544
Agent 4 (E — most downstream):
11
0.0
1

CHOOSER
8
544
148
589
agent4-backend
agent4-backend
"anthropic" "openai" "ollama" "google"
0

INPUTBOX
150
544
230
589
agent4-model
claude-sonnet-4-6
1
0
String

INPUTBOX
8
594
460
654
ollama-base-url
http://localhost:11434/v1
1
0
String

INPUTBOX
8
658
460
748
system-prompt-override
 
1
0
String

MONITOR
471
415
571
460
public good (pg)
pg
0
1
11

MONITOR
575
415
675
460
mean contribution
mean-contribution
2
1
11

MONITOR
679
415
779
460
mean extraction
mean-extraction
2
1
11

MONITOR
783
415
871
460
institution score
institution-score
0
1
11

PLOT
471
465
871
665
Contributions per agent (A-E)
Round
Tokens invested
0.0
10.0
0.0
10.0
true
true
"" ""
PENS
"A (upstream)" 1.0 0 -13345367 true "" "plot [contribution] of turtle 0"
"B" 1.0 0 -10899396 true "" "plot [contribution] of turtle 1"
"C (middle)" 1.0 0 -2674135 true "" "plot [contribution] of turtle 2"
"D" 1.0 0 -955883 true "" "plot [contribution] of turtle 3"
"E (downstream)" 1.0 0 -5825686 true "" "plot [contribution] of turtle 4"

PLOT
471
668
871
868
Extractions per agent (A-E)
Round
Water collected
0.0
10.0
0.0
100.0
true
true
"" ""
PENS
"A (upstream)" 1.0 0 -13345367 true "" "plot [collected] of turtle 0"
"B" 1.0 0 -10899396 true "" "plot [collected] of turtle 1"
"C (middle)" 1.0 0 -2674135 true "" "plot [collected] of turtle 2"
"D" 1.0 0 -955883 true "" "plot [collected] of turtle 3"
"E (downstream)" 1.0 0 -5825686 true "" "plot [collected] of turtle 4"

PLOT
471
870
871
1000
Public infrastructure (water)
Round
Water available
0.0
10.0
0.0
100.0
true
false
"" ""
PENS
"pg" 1.0 0 -16777216 true "" "plot pg"

PLOT
471
1003
871
1133
Income per agent (A-E)
Round
Income
0.0
10.0
0.0
30.0
true
true
"" ""
PENS
"A (upstream)" 1.0 0 -13345367 true "" "plot [income] of turtle 0"
"B" 1.0 0 -10899396 true "" "plot [income] of turtle 1"
"C (middle)" 1.0 0 -2674135 true "" "plot [income] of turtle 2"
"D" 1.0 0 -955883 true "" "plot [income] of turtle 3"
"E (downstream)" 1.0 0 -5825686 true "" "plot [income] of turtle 4"

@#$#@#$#@
Irrigation Game — LLM Edition

An LLM-augmented adaptation of Janssen (2012) "An Agent-based Model based on Field Experiments."

Five farmers (positions A–E, upstream to downstream) manage a shared irrigation canal.
Each round:
1. CONTRIBUTION: All farmers simultaneously invest 0–10 tokens in infrastructure.
   Total investment determines water flow (nonlinear step function).
2. EXTRACTION: Farmers extract water sequentially upstream→downstream.

Original model: decision-theoretic (Fehr-Schmidt utility, learning).
This version: LLM agents reason in natural language about cooperation and fairness.

See irrigation_llm_bridge.py and README.md for details.
@#$#@#$#@
default
true
0
Polygon -7500403 true true 150 5 40 250 150 205 260 250

person
false
0
Circle -7500403 true true 110 5 80
Polygon -7500403 true true 105 90 120 195 90 285 105 300 135 300 150 225 165 300 195 300 210 285 180 195 195 90
Rectangle -7500403 true true 127 79 172 94
Polygon -7500403 true true 195 90 240 150 225 180 165 105
Polygon -7500403 true true 105 90 60 150 75 180 135 105

@#$#@#$#@
NetLogo 7.0.0
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
default
0.0
-0.2 0 0.0 1.0
0.0 1 1.0 0.0
0.2 0 0.0 1.0
link direction
true
0
Line -7500403 true 150 150 90 180
Line -7500403 true 150 150 210 180

@#$#@#$#@
@#$#@#$#@
@#$#@#$#@
