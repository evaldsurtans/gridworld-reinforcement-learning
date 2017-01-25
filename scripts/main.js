var const_max_obstacles = 8;
var const_env_width = 6;
var const_env_height = 7;
var const_value_terminal = 10.0;

var const_gamma_discount = 0.9;
var const_training_data_sequences = 50;

var const_ahead = 1;
var const_left = 2;
var const_right = 3;
var const_back = 4;
var const_probabilties = [0, 0.7, 0.14, 0.14, 0.02]; //ahead, left, right, back
var const_use_probabilties = true;
var const_td_learning_rate = 0.5;

var const_up = 1;
var const_down = 4;

var values = null;
var known_values = null;
var obstacles = null;
var policies = null;
var values_source = null;
var values_iterations = null;

var obstacles_positions = null;
var values_positions = null;
var known_values_positions = null;

var training_samples = [];
var training_samples_actions = [];
var training_samples_map = null;
var value_intial = 0;
var value_intial_prev = -1;
var temp_policies = [];

var graph_trace = {
    x: [],
    y: [],
    type: 'scatter'
};

function display_environment() {
	var jEnv = $('#env');
	jEnv.children().remove();

	for (var j = 0; j < const_env_height; j++) {
		for (var i = 0; i < const_env_width; i++) {
			cell_type = obstacles.get(i, j) ? 'obstacle' : '';

			if (!obstacles.get(i, j)) {
				if (values.get(i, j) == const_value_terminal)
					cell_type = 'good';
				else if (values.get(i, j) == -const_value_terminal)
					cell_type = 'bad';
				else if (training_samples_map.get(i, j))
					cell_type = 'training';
			}

			var jCell = $('<div class="cell ' + cell_type + '"></div>');
			jCell.appendTo(jEnv);

			if(policies.get(i, j) ||
                known_values.get(i, j)) {
                jCell.append('<div class="value">' + values.get(i, j).toFixed(4) + '</div>');
            }

			if (policies.get(i, j)) {

				var arrow = '';
				switch (policies.get(i, j)) {
					case const_up:
						arrow = '&#x2191;';
						break;
					case const_down:
						arrow = '&#x2193;';
						break;
					case const_left:
						arrow = '&#x2190;';
						break;
					case const_right:
						arrow = '&#x2192;';
						break;
				}
				jCell.append('<div class="arrow">' + arrow + '</div>');
			}
		}
		jEnv.append('<div class="clear"></div>');
	}

    display_graph();
}

function is_valid_state(x, y) {
	return (x >= 0 && x < const_env_width &&
		y >= 0 && y < const_env_height &&
		!obstacles.get(x, y));
}

function is_policy_valid(x, y, prev) {
	var policy = policies.get(x, y);

	if (policy == 0) return true;

	if (known_values.get(x, y))
		return true;

	for (var i = 0; i < prev.length; i++) {
		if (prev[i][0] == x && prev[i][1] == y) {
			//Loop detected
			return false;
		}
	}

	var next_state = [x, y];
	switch (policy) {
		case const_up:
			next_state[1] -= 1;
			break;
		case const_down:
			next_state[1] += 1;
			break;
		case const_left:
			next_state[0] -= 1;
			break;
		case const_right:
			next_state[0] += 1;
			break;
	}

	prev.push([x, y]);

	return is_policy_valid(next_state[0], next_state[1], prev);
}

function get_path(x, y, prev) {
	var policy = policies.get(x, y);

	if (known_values.get(x, y)) {
		prev.push([x, y]);
		return prev;
	}

	for (var i = 0; i < prev.length; i++) {
		if (prev[i][0] == x && prev[i][1] == y) {
			//Loop detected
			return prev;
		}
	}

	if (policy == 0 || policy == undefined) return prev;

	prev.push([x, y]);

	var actions_available = [];
	var next_state = [x, y];

	if(is_valid_state(x - 1, y)) {
		actions_available.push(const_left);
	}
	if(is_valid_state(x + 1, y)) {
		actions_available.push(const_right);
	}
	if(is_valid_state(x, y - 1)) {
		actions_available.push(const_up);
	}
	if(is_valid_state(x, y + 1)) {
		actions_available.push(const_down);
	}
	
	if(const_use_probabilties)
	{
		policy = get_stohastic_action(policy, actions_available);
	}

	temp_policies.push(policy);

	switch (policy) {
		case const_up:
			next_state[1] -= 1;
			break;
		case const_down:
			next_state[1] += 1;
			break;
		case const_left:
			next_state[0] -= 1;
			break;
		case const_right:
			next_state[0] += 1;
			break;
	}

	return get_path(next_state[0], next_state[1], prev);
}

function get_stohastic_action(action_desired, actions_available) {
	var weights = [];
	var actions = [];

	if(action_desired == const_up) {
		if(actions_available.indexOf(const_up) >= 0) {
			actions.push(const_up);
			weights.push(const_probabilties[const_ahead])
		}
		if(actions_available.indexOf(const_left) >= 0) {
			actions.push(const_left);
			weights.push(const_probabilties[const_left])
		}
		if(actions_available.indexOf(const_right) >= 0) {
			actions.push(const_right);
			weights.push(const_probabilties[const_right])
		}
		if(actions_available.indexOf(const_down) >= 0) {
			actions.push(const_down);
			weights.push(const_probabilties[const_back])
		}
	}
	else if(action_desired == const_down) {
		if(actions_available.indexOf(const_down) >= 0) {
			actions.push(const_down);
			weights.push(const_probabilties[const_ahead])
		}
		if(actions_available.indexOf(const_left) >= 0) {
			actions.push(const_left);
			weights.push(const_probabilties[const_right])
		}
		if(actions_available.indexOf(const_right) >= 0) {
			actions.push(const_right);
			weights.push(const_probabilties[const_left])
		}
		if(actions_available.indexOf(const_up) >= 0) {
			actions.push(const_up);
			weights.push(const_probabilties[const_back])
		}
	}
	else if(action_desired == const_left) {
		if(actions_available.indexOf(const_left) >= 0) {
			actions.push(const_left);
			weights.push(const_probabilties[const_ahead])
		}
		if(actions_available.indexOf(const_up) >= 0) {
			actions.push(const_up);
			weights.push(const_probabilties[const_right])
		}
		if(actions_available.indexOf(const_down) >= 0) {
			actions.push(const_down);
			weights.push(const_probabilties[const_left])
		}
		if(actions_available.indexOf(const_right) >= 0) {
			actions.push(const_right);
			weights.push(const_probabilties[const_back])
		}
	}
	else if(action_desired == const_right) {
		if(actions_available.indexOf(const_right) >= 0) {
			actions.push(const_right);
			weights.push(const_probabilties[const_ahead])
		}
		if(actions_available.indexOf(const_up) >= 0) {
			actions.push(const_up);
			weights.push(const_probabilties[const_left])
		}
		if(actions_available.indexOf(const_down) >= 0) {
			actions.push(const_down);
			weights.push(const_probabilties[const_right])
		}
		if(actions_available.indexOf(const_left) >= 0) {
			actions.push(const_left);
			weights.push(const_probabilties[const_back])
		}
	}

	var rand = nj.array(weights).sum() * Math.random();
	var tmp = 0;
	var result = actions[0];
	for(var i = 0; i < weights.length; i++)
	{
		tmp += weights[i];
		if(rand <= tmp)
		{
			result = actions[i];
			break;
		}
	}

	//console.log('desired ' + action_desired.toString() + ' actual: ' + result.toString());

	return result;
}

function calc_value_iterations() {
	var watchdog = 0;
	var is_policies_determined = false;
	while (!is_policies_determined && watchdog++ < 10000) {
		for (var j = 0; j < values_positions.length; j++) {
			var pos = values_positions[j];
			var x = pos[0],
				y = pos[1];

			if (is_valid_state(x, y) && !known_values.get(x, y)) {
				max_value_next = -Number.MAX_VALUE;

				if (is_valid_state(x - 1, y)) {
					value_next = const_probabilties[const_ahead] * values.get(x - 1, y);
					if (is_valid_state(x, y - 1)) {
						value_next += const_probabilties[const_right] * values.get(x, y - 1);
					}
					if (is_valid_state(x, y + 1)) {
						value_next += const_probabilties[const_left] * values.get(x, y + 1);
					}
					if (is_valid_state(x + 1, y)) {
						value_next += const_probabilties[const_back] * values.get(x, y + 1);
					}

					max_value_next = Math.max(value_next, max_value_next);
				}

				if (is_valid_state(x + 1, y)) {
					value_next = const_probabilties[const_ahead] * values.get(x + 1, y);
					if (is_valid_state(x, y + 1)) {
						value_next += const_probabilties[const_right] * values.get(x, y + 1);
					}
					if (is_valid_state(x, y - 1)) {
						value_next += const_probabilties[const_left] * values.get(x, y - 1);
					}
					if (is_valid_state(x - 1, y)) {
						value_next += const_probabilties[const_back] * values.get(x - 1, y);
					}

					max_value_next = Math.max(value_next, max_value_next);
				}

				if (is_valid_state(x, y + 1)) {
					value_next = const_probabilties[const_ahead] * values.get(x, y + 1);
					if (is_valid_state(x - 1, y)) {
						value_next += const_probabilties[const_right] * values.get(x - 1, y);
					}
					if (is_valid_state(x + 1, y)) {
						value_next += const_probabilties[const_left] * values.get(x + 1, y);
					}
					if (is_valid_state(x, y - 1)) {
						value_next += const_probabilties[const_back] * values.get(x, y - 1);
					}

					max_value_next = Math.max(value_next, max_value_next);
				}

				if (is_valid_state(x, y - 1)) {
					value_next = const_probabilties[const_ahead] * values.get(x, y - 1);
					if (is_valid_state(x + 1, y)) {
						value_next += const_probabilties[const_right] * values.get(x + 1, y);
					}
					if (is_valid_state(x - 1, y)) {
						value_next += const_probabilties[const_left] * values.get(x - 1, y);
					}
					if (is_valid_state(x, y + 1)) {
						value_next += const_probabilties[const_back] * values.get(x, y + 1);
					}

					max_value_next = Math.max(value_next, max_value_next);
				}

				value = value_intial + const_gamma_discount * max_value_next;

				values.set(x, y, value);
			}
		}

		//Determine policies
		is_policies_determined = evaluate_policies(true);

		if (is_policies_determined) {
			for (var j = 0; j < values_positions.length; j++) {
				var pos = values_positions[j];
				var x = pos[0],
					y = pos[1];

				if (!is_policy_valid(x, y, [])) {
					is_policies_determined = false;
					break;
				}
			}
		}

        add_graph_iteration();

	} //iterations

	console.log('iterations needed: ', watchdog);

    values_iterations = values.clone();

}

function evaluate_policies(is_break, is_use_training_data) {
	var is_policies_determined = true;
	for (var j = 0; j < values_positions.length; j++) {
		var pos = values_positions[j];
		var x = pos[0],
			y = pos[1];

		if (known_values.get(x, y))
			continue;

		var value = values.get(x, y);
		var max_value = -Number.MAX_VALUE;
		var max_direction = 0;

		var arr_vectors = [
			[-1, 0],
			[1, 0],
			[0, 1],
			[0, -1]
		];
		var arr_directions = [const_left, const_right, const_down, const_up];

		for (var z = 0; z < arr_directions.length; z++) {
			var pos_next = [x + arr_vectors[z][0], y + arr_vectors[z][1]];
			if (is_valid_state(pos_next[0], pos_next[1])) {

				if(!is_use_training_data ||
				(is_use_training_data && training_samples_map.get(pos_next[0], pos_next[1]))) {
					value_next = values.get(pos_next[0], pos_next[1]);
					if (value == value_next && is_break) { //Not clearly defined policies
						is_policies_determined = false;
						break;
					} else if (max_value < value_next) {
						max_value = value_next;
						max_direction = arr_directions[z];
					}
				}
			}
		}

		if (!is_policies_determined && is_break) {
			//console.log('policies do not converge');
			break;
		}

		if(!is_use_training_data ||
			(is_use_training_data && training_samples_map.get(x,y)))
		{
			policies.set(x, y, max_direction);
		}
	}

	return is_policies_determined;
}

function generate_traning_data() {

	if (value_intial != value_intial_prev ||
        training_samples.length == 0) {

		training_samples = [];
		training_samples_actions = [];
		training_samples_map = nj.zeros([const_env_width, const_env_height]);

		value_intial_prev = value_intial;
		var used_postions = values_positions.slice(0, values_positions.length - known_values_positions.length);

		//Remove non path nodes
		var i = 0;
		while (i < used_postions.length) {
			var path = get_path(used_postions[i][0], used_postions[i][1], []);
			if (path.length < 2) {
				used_postions.splice(i, 1);
			} else {
				i++;
			}
		}

		//Choose random training data
		for (var i = 0; i < const_training_data_sequences && used_postions.length > 0; i++) {
			var index = Math.round((used_postions.length - 1) * Math.random());

			temp_policies = [];
			var states = get_path(used_postions[index][0], used_postions[index][1], []);

			training_samples.push(states);
			training_samples_actions.push(temp_policies);

			used_postions.splice(index, 1);
		}

		var jTraining = $('#training');
		jTraining.children().remove();
		jTraining = $('<div/>').appendTo(jTraining);

		i = 1;
		training_samples.forEach(function(sample) {
			jTraining.append((i++).toString() + '. ' + JSON.stringify(sample) + '<br/>');

			sample.forEach(function(state) {
				training_samples_map.set(state[0], state[1], 1);
			});
		});
	}
}

function initialize_values() {
	values = nj.ones([const_env_width, const_env_height]); //Main Value matrix
	value_intial = parseFloat($('#cost-state').val());
	values = values.multiply(value_intial);

	policies = nj.zeros([const_env_width, const_env_height]);

	var pos = known_values_positions[0];
	values.set(pos[0], pos[1], const_value_terminal);

	var pos = known_values_positions[1];
	values.set(pos[0], pos[1], -const_value_terminal);

    values_source = values.clone();

    graph_trace.x = [];
    graph_trace.y = [];
}


function get_action_by_diff(state, state_next)
{
	var diff_x = state_next[0] - state[0];
	var diff_y = state_next[1] - state[1];
	if(diff_y > 0)
		return const_down;
	if(diff_x > 0)
		return const_right;
	if(diff_x < 0)
		return const_left;

	return const_up;
}

function map_load() {
    var serialized = JSON.parse($('#input-saved-state').val());
    obstacles = nj.array(serialized.obstacles);
    known_values = nj.array(serialized.known_values);
    obstacles_positions = serialized.obstacles_positions;
    values_positions = serialized.values_positions;
    known_values_positions = serialized.known_values_positions;
    training_samples_map = serialized.training_samples_map;
    training_samples_actions = serialized.training_samples_actions;
    training_samples = serialized.training_samples;

    value_iteration();
}

function map_generate() {
    obstacles = null;
    training_samples = [];
    value_iteration();
}

function setCookie(cname, cvalue, exdays) {
    var d = new Date();
    d.setTime(d.getTime() + (exdays * 24 * 60 * 60 * 1000));
    var expires = "expires="+d.toUTCString();
    document.cookie = cname + "=" + cvalue + ";" + expires + ";path=/";
}

function getCookie(cname) {
    var name = cname + "=";
    var ca = document.cookie.split(';');
    for(var i = 0; i < ca.length; i++) {
        var c = ca[i];
        while (c.charAt(0) == ' ') {
            c = c.substring(1);
        }
        if (c.indexOf(name) == 0) {
            return c.substring(name.length, c.length);
        }
    }
    return "";
}

function value_iteration() {

	if (obstacles == null) { //Reset env
		obstacles = nj.zeros([const_env_width, const_env_height]);
		known_values = nj.zeros([const_env_width, const_env_height]); //Initial values

		obstacles_positions = [];
		values_positions = [];
		known_values_positions = [];

		for (var j = 0; j < const_env_height; j++) {
			for (var i = 0; i < const_env_width; i++) {
				values_positions.push([i, j]);
			}
		}

		//Obstacles
		for (var i = 0; i < const_max_obstacles; i++) {
			var pos = values_positions[Math.round((values_positions.length - 1) * Math.random())];
			var x = pos[0],
				y = pos[1];
			obstacles.set(x, y, 1.0);
			obstacles_positions.push([x, y]);
			values_positions.splice(values_positions.indexOf(pos), 1);
		}

		//Add initial states
		var pos = values_positions[Math.round((values_positions.length - 1) * Math.random())];
		known_values.set(pos[0], pos[1], 1);
		known_values_positions.push(pos);
		values_positions.splice(values_positions.indexOf(pos), 1);
		console.log('good', pos);

		pos = values_positions[Math.round((values_positions.length - 1) * Math.random())];
		known_values.set(pos[0], pos[1], 1);
		known_values_positions.push(pos);
		values_positions.splice(values_positions.indexOf(pos), 1);
		console.log('bad', pos);

		values_positions = values_positions.concat(known_values_positions);

	}

	initialize_values();

	calc_value_iterations();

	generate_traning_data();

	display_environment();


    $('#input-saved-state').val(JSON.stringify({
        obstacles: obstacles.tolist(),
        known_values: known_values.tolist(),
        obstacles_positions: obstacles_positions,
        values_positions: values_positions,
        known_values_positions: known_values_positions,
        training_samples_map : training_samples_map,
        training_samples_actions : training_samples_actions,
        training_samples : training_samples
    }));

    setCookie('state', $('#input-saved-state').val(), 365);
}

//Direct value estimation
function train_de() {
	initialize_values();
    var rewards = values.clone();

	model_rewards = nj.zeros([const_env_width, const_env_height]);
	model_rewards_counts = nj.zeros([const_env_width, const_env_height]);

	training_samples.forEach(function(sample) {
		sample_reversed = sample.slice().reverse();

		total = 0;
		sample_reversed.forEach(function(state) {
			total = rewards.get(state[0], state[1]) + const_gamma_discount * total;

			model_rewards.set(state[0], state[1], model_rewards.get(state[0], state[1]) + total);
			model_rewards_counts.set(state[0], state[1], model_rewards_counts.get(state[0], state[1]) + 1.0);
		});

        for (var j = 0; j < const_env_height; j++) {
            for (var i = 0; i < const_env_width; i++) {
                if (model_rewards_counts.get(i, j) > 0) {
                    if(known_values.get(i, j) == 0)
                    {
                        values.set(i, j, model_rewards.get(i, j) / model_rewards_counts.get(i, j));
                    }
                }
            }
        }

        add_graph_iteration();

	});

	evaluate_policies(false, true);
	display_environment();
}

//Adaptive Dynamic Program
function train_adp() {
	initialize_values();

    var rewards = values.clone();

	model_visited = nj.zeros([const_env_width, const_env_height]);
	model_U = nj.zeros([const_env_width, const_env_height]);
	model_R = nj.zeros([const_env_width, const_env_height]);

	//State action (trying to execute action)
	model_N_SA = nj.zeros([const_env_width, const_env_height, 5]);
	//State action state (succeed to execute action)
	model_N_SAS = nj.zeros([const_env_width, const_env_height, 5, 5]);


	training_samples.forEach(function(states, index_training) {
		
		states_reversed = states.slice().reverse();
		actions_reversed = training_samples_actions[index_training].slice().reverse();

		states_reversed.forEach(function(state, index_state) {
			
			if (!model_visited.get(state[0], state[1])) {
				var value = rewards.get(state[0], state[1]);

				model_U.set(state[0], state[1], value);
				model_R.set(state[0], state[1], value);
			}

			model_visited.set(state[0], state[1], model_visited.get(state[0], state[1]) + 1);

			//Not terminal state
			if(index_state > 0)
			{
				var state_next = states_reversed[index_state - 1];
				var action_real = get_action_by_diff(state, state_next);
				var action_planned = actions_reversed[index_state - 1];
				
				model_N_SAS.set(state[0], state[1], action_planned, action_real, model_N_SAS.get(state[0], state[1], action_planned, action_real) + 1.0);
				model_N_SA.set(state[0], state[1], action_planned, model_N_SA.get(state[0], state[1], action_planned) + 1.0);

				//console.log('--');

				sum = 0.0;
				for(var action = 1; action <= 4; action++) 
				{
					if(model_N_SA.get(state[0], state[1], action_planned) == 0 ||
						model_N_SA.get(state[0], state[1], action_planned) == undefined) continue;

					var P = model_N_SAS.get(state[0], state[1], action_planned, action) / model_N_SA.get(state[0], state[1], action_planned);
					var U_a = 0;
					switch(action)
					{
						case const_up:
							if(is_valid_state(state[0], state[1] - 1))
							{
								U_a = model_U.get(state[0], state[1] - 1)
							}
							break;
						case const_down:
							if(is_valid_state(state[0], state[1] + 1))
							{
								U_a = model_U.get(state[0], state[1] + 1)
							}
							break;
						case const_left:
							if(is_valid_state(state[0] - 1, state[1]))
							{
								U_a = model_U.get(state[0] - 1, state[1])
							}
							break;
						case const_right:
							if(is_valid_state(state[0] + 1, state[1]))
							{
								U_a = model_U.get(state[0] + 1, state[1])
							}
							break;
					}

					sum += P * U_a;

					//console.log(P, U_a);
				}

				var U = model_R.get(state[0], state[1]) + const_gamma_discount * sum;
				model_U.set(state[0], state[1], U);
			}
		});

        for (var j = 0; j < const_env_height; j++) {
            for (var i = 0; i < const_env_width; i++) {
                if (model_visited.get(i, j) > 0) {
                    if(known_values.get(i, j) == 0)
                    {
                        values.set(i, j, model_U.get(i, j));
                    }
                }
            }
        }

        add_graph_iteration();

	});

	evaluate_policies(false, true);
	display_environment();
}

function train_td() {
    initialize_values();

    var rewards = values.clone();

    training_samples.forEach(function(states, index_training) {

        states_reversed = states.slice().reverse();
        states_reversed.forEach(function(state, index_state) {
            var x = state[0], y = state[1];

            //Not terminal state
            if(index_state > 0)
            {
                var U = values.get(x, y);
                var R = rewards.get(x, y);

                var S_next = states_reversed[index_state-1];
                var x_next = S_next[0], y_next = S_next[1];
                var U_next = values.get(x_next, y_next);

                U = U + const_td_learning_rate * (R + const_gamma_discount * U_next - U);

                values.set(x, y, U);
            }
        });

        add_graph_iteration();

    });

    evaluate_policies(false, true);
    display_environment();
}

function add_graph_iteration() {
    if(values_iterations != null) {
        var mse = 0;
        for (var j = 0; j < const_env_height; j++) {
            for (var i = 0; i < const_env_width; i++) {
                mse += Math.pow(values_iterations.get(i, j) - values.get(i, j), 2);
            }
        }
        mse /= (const_env_height * const_env_width);

        graph_trace.x.push(graph_trace.x.length + 1);
        graph_trace.y.push(mse);
    }
}

function display_graph() {

    var data = [graph_trace];

    var layout = {
        xaxis: {
            title: 'Samples / Iterations'
        },
        yaxis: {
            title: 'MSE'
        }
    };


    Plotly.newPlot('graph', data, layout);
}

$(document).ready(function() {
	$('#button-value-iteration').click(value_iteration);
	$('#button-direct-estimate').click(train_de);
	$('#button-adp').click(train_adp);
	$('#button-td').click(train_td);

    $('#button-generate').click(map_generate);
    $('#button-restore').click(map_load);

    if(getCookie("state").length > 0) {
        $('#input-saved-state').val(getCookie("state"));
        map_load();
    }
    else {
        value_iteration();
    }

    display_graph();

});