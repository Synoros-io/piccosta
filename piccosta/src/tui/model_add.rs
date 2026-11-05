// A few ordinary pickers and text inputs; the secret lives here, never in a session or draft.
use super::{App, Picker, Update};
use crate::model_add::{Addition, Probe, APIS};

impl App {
    fn model_input(&mut self, title: &'static str, initial: String, masked: bool, apply: fn(&mut App, String)) {
        self.picker = Some(Picker { title: title.to_string(), answer: None, items: Vec::new(), filter: initial, selected: 0, apply, delete: None, confirming: false, free: None, text: true, masked });
    }

    pub(super) fn add_model(&mut self) {
        self.model_flow += 1;
        self.model_add = None;
        self.model_input("add models · base URL · esc cancels", String::new(), false, |app, base| {
            match Addition::new(&base) {
                Ok(addition) => { app.model_add = Some(addition); app.probe_model(); }
                Err(error) => app.notice = Some(error),
            }
        });
    }

    fn probe_model(&mut self) {
        let Some(mut addition) = self.model_add.take() else { return };
        let (sender, agent_dir, flow) = (self.sender.clone(), self.host.agent_dir.clone(), self.model_flow);
        self.model_input("Probing endpoint… · esc cancels", String::new(), false, |_, _| {});
        self.model_probe = Some(tokio::spawn(async move {
            let result = addition.probe(&agent_dir).await.map(|probe| (addition, probe));
            let _ = sender.send((0, Update::ModelProbe(flow, result)));
        }).abort_handle());
    }

    pub(super) fn probed_model(&mut self, flow: u64, result: Result<(Addition, Probe), String>) {
        if flow != self.model_flow { return; }
        self.model_probe = None;
        self.picker = None;
        match result {
            Err(error) => self.notice = Some(error),
            Ok((addition, probe)) => {
                self.model_add = Some(addition);
                match probe {
                    Probe::KeyNeeded => self.pick("server needs a key · choose its source", vec!["paste key (hidden)".into(), "environment variable".into(), "command".into()], "", |app, source| {
                        let literal = source.starts_with("paste");
                        let prefix = if literal { "" } else if source == "environment variable" { "$" } else { "!" };
                        if let Some(addition) = &mut app.model_add { addition.literal = literal; addition.key = prefix.into(); }
                        app.model_input(if literal { "paste key · hidden · esc cancels" } else if prefix == "$" { "environment variable name" } else { "command whose output is the key" }, String::new(), true, |app, key| {
                            if let Some(addition) = &mut app.model_add { addition.key.push_str(&key); }
                            app.probe_model();
                        });
                    }),
                    Probe::Ready => {
                        let api = self.model_add.as_ref().map(|addition| addition.api.clone()).unwrap_or_default();
                        self.pick("API format · enter keeps detected format", APIS.iter().map(|api| api.to_string()).collect(), "", |app, api| {
                            let Some(addition) = &mut app.model_add else { return };
                            addition.api = api;
                            let name = addition.name.clone();
                            app.model_input("provider name · edit or enter keeps it", name, false, |app, name| {
                                let Some(addition) = &mut app.model_add else { return };
                                addition.name = name;
                                if addition.manual {
                                    app.model_input("no model list · enter a model id", String::new(), false, |app, id| {
                                        if let Some(addition) = &mut app.model_add { addition.models = vec![id]; }
                                        app.save_model();
                                    });
                                } else {
                                    let models = addition.models.clone();
                                    app.pick("models found · enter adds provider", models, "", |app, _| app.save_model());
                                }
                            });
                        });
                        self.select(&api);
                    }
                }
            }
        }
    }

    fn save_model(&mut self) {
        let Some(addition) = self.model_add.take() else { return };
        let (sender, agent_dir) = (self.sender.clone(), self.host.agent_dir.clone());
        tokio::spawn(async move {
            let result = addition.save(&agent_dir).await.unwrap_or_else(|error| error);
            let _ = sender.send((0, Update::Report(result)));
        });
    }
}
