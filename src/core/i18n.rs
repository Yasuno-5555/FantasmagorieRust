//! Localization (I18n) Manager
//! Ported from i18n.hpp

use std::collections::HashMap;
use std::cell::RefCell;

thread_local! {
    static INSTANCE: RefCell<I18nManager> = RefCell::new(I18nManager::new());
}

pub struct I18nManager {
    pub current_locale: String,
    pub catalogs: HashMap<String, HashMap<String, String>>,
}

impl I18nManager {
    pub fn new() -> Self {
        Self {
            current_locale: "en".to_string(),
            catalogs: HashMap::new(),
        }
    }

    pub fn set_locale(locale: &str) {
        INSTANCE.with(|i| i.borrow_mut().current_locale = locale.to_string());
    }

    pub fn add_translation(locale: &str, key: &str, value: &str) {
        INSTANCE.with(|i| {
            let mut i = i.borrow_mut();
            let cat = i.catalogs.entry(locale.to_string()).or_insert_with(HashMap::new);
            cat.insert(key.to_string(), value.to_string());
        });
    }

    pub fn t(key: &str) -> String {
        INSTANCE.with(|i| {
            let i = i.borrow();
            if let Some(cat) = i.catalogs.get(&i.current_locale) {
                if let Some(val) = cat.get(key) {
                    return val.clone();
                }
            }
            key.to_string()
        })
    }
    
    /// Translation with placeholders: t("Hello {name}", {"name": "World"})
    pub fn tf(key: &str, args: &HashMap<String, String>) -> String {
        let mut text = Self::t(key);
        for (k, v) in args {
            text = text.replace(&format!("{{{}}}", k), v);
        }
        text
    }
}
