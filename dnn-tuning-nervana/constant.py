TUNABLE_PARAMS = [
            {
                "bounds": {
                    "max": -0.3,
                    "min": -3.0,
                    },
                "name": "log(learning_rate)",
                "type": "double",
                },
            {
                "bounds": {
                    "max": 0.0,
                    "min": -3.0,
                    },
                "name": "log(weight_decay)",
                "type": "double",
                },
            {
                "bounds": {
                    "max": 0.5,
                    "min": 0.01,
                    },
                "name": "gaussian_scale",
                "type": "double",
                },
            {
                "bounds": {
                    "max": 0.999,
                    "min": 0.001,
                    },
                "name": "momentum_coef",
                "type": "double",
                },
            {
                "bounds": {
                    "max": 0.999,
                    "min": 0.001,
                    },
                "name": "momentum_step_change",
                "type": "double",
                },
            {
                "bounds": {
                    "max": 300,
                    "min": 50,
                    },
                "name": "momentum_step_schedule_start",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 100,
                    "min": 5,
                    },
                "name": "momentum_step_schedule_step_width",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 20,
                    "min": 1,
                    },
                "name": "momentum_step_schedule_steps",
                "type": "int",
                },
            {
                "bounds": {
                    "max": 500,
                    "min": 50,
                    },
                "name": "epochs",
                "type": "int",
                },
            ]

PAPER_PARAMS = {
    "log(learning_rate)": -1.30103,
    "log(weight_decay)": -3.0,
    "gaussian_scale": 0.05,
    "momentum_coef": 0.9,
    "momentum_step_change": 0.1,
    "momentum_step_schedule_start": 200,
    "momentum_step_schedule_step_width": 50,
    "momentum_step_schedule_steps": 3,
    "epochs": 350,
    }
