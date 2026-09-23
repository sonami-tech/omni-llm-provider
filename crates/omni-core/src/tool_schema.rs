//! Tool schema validation and provider-specific wire shaping.

use std::collections::{BTreeMap, BTreeSet};

use serde_json::{Map, Value, json};

const COMBINATORS: [&str; 3] = ["oneOf", "anyOf", "allOf"];

fn combinator(schema: &Map<String, Value>) -> Result<Option<(&'static str, &[Value])>, String> {
    let mut found = None;
    for key in COMBINATORS {
        if let Some(value) = schema.get(key) {
            let branches = value
                .as_array()
                .filter(|a| !a.is_empty())
                .ok_or_else(|| format!("{key} must be a non-empty array"))?;
            if found.is_some() {
                return Err("a tool schema cannot have multiple sibling combinators".into());
            }
            found = Some((key, branches.as_slice()));
        }
    }
    Ok(found)
}

fn schema_children(obj: &Map<String, Value>) -> Vec<&Value> {
    let mut children = Vec::new();
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ] {
        if let Some(entries) = obj.get(key).and_then(Value::as_object) {
            children.extend(entries.values());
        }
    }
    for key in [
        "items",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
    ] {
        if let Some(value) = obj.get(key) {
            if let Some(items) = value.as_array() {
                children.extend(items);
            } else {
                children.push(value);
            }
        }
    }
    for key in ["prefixItems", "oneOf", "anyOf", "allOf"] {
        if let Some(values) = obj.get(key).and_then(Value::as_array) {
            children.extend(values.iter());
        }
    }
    children
}

fn validate_arrays(value: &Value) -> Result<(), String> {
    if let Value::Object(obj) = value {
        for key in COMBINATORS {
            if obj
                .get(key)
                .is_some_and(|v| v.as_array().is_none_or(Vec::is_empty))
            {
                return Err(format!("{key} must be a non-empty array"));
            }
        }
        for child in schema_children(obj) {
            validate_arrays(child)?;
        }
    }
    Ok(())
}

fn contains_key(value: &Value, key: &str) -> bool {
    if let Value::Object(obj) = value {
        obj.contains_key(key) || schema_children(obj).iter().any(|v| contains_key(v, key))
    } else {
        false
    }
}

/// Validate the inbound protocol without changing the client schema.
pub fn validate_tool_schema(schema: &Value, strict: bool, anthropic: bool) -> Result<(), String> {
    validate_arrays(schema)?;
    let root = schema.as_object();
    if anthropic {
        let root = root.ok_or("input_schema must be an object")?;
        if COMBINATORS.iter().any(|key| root.contains_key(*key)) {
            return Err("Anthropic input_schema cannot have a root combinator".into());
        }
        if strict && (contains_key(schema, "oneOf") || contains_allof_ref(schema)) {
            return Err("Anthropic strict tools cannot use oneOf or allOf with $ref".into());
        }
    } else if strict
        && (root.and_then(|o| o.get("type")).and_then(Value::as_str) != Some("object")
            || root.is_some_and(|o| COMBINATORS.iter().any(|key| o.contains_key(*key)))
            || contains_key(schema, "oneOf")
            || contains_key(schema, "allOf"))
    {
        return Err(
            "strict tools require a root object without root combinators, oneOf, or allOf".into(),
        );
    }
    Ok(())
}

fn contains_allof_ref(value: &Value) -> bool {
    if let Value::Object(obj) = value {
        obj.get("allOf")
            .and_then(Value::as_array)
            .is_some_and(|branches| branches.iter().any(|b| contains_key(b, "$ref")))
            || schema_children(obj).iter().any(|v| contains_allof_ref(v))
    } else {
        false
    }
}

fn object_rules(value: &Value, required_properties: bool) -> bool {
    let Value::Object(obj) = value else {
        return true;
    };
    let is_object = obj.get("type").is_some_and(|t| {
        t == "object"
            || t.as_array()
                .is_some_and(|types| types.iter().any(|v| v == "object"))
    }) || obj.contains_key("properties");
    if is_object {
        if obj.get("additionalProperties") != Some(&Value::Bool(false)) {
            return false;
        }
        if required_properties {
            let required: BTreeSet<&str> = obj
                .get("required")
                .and_then(Value::as_array)
                .into_iter()
                .flatten()
                .filter_map(Value::as_str)
                .collect();
            if obj
                .get("properties")
                .and_then(Value::as_object)
                .is_some_and(|p| p.keys().any(|key| !required.contains(key.as_str())))
            {
                return false;
            }
        }
    }
    schema_children(obj)
        .iter()
        .all(|child| object_rules(child, required_properties))
}

pub fn claude_strict(schema: &Value, requested: bool) -> bool {
    requested && !contains_key(schema, "oneOf") && object_rules(schema, false)
}

pub fn codex_strict(schema: &Value, requested: bool) -> bool {
    requested
        && schema.get("type").and_then(Value::as_str) == Some("object")
        && !COMBINATORS.iter().any(|key| schema.get(key).is_some())
        && !contains_key(schema, "oneOf")
        && !contains_key(schema, "allOf")
        && object_rules(schema, true)
}

fn validate_branch_tree(schema: &Map<String, Value>) -> Result<(), String> {
    if schema.get("type").is_some_and(|t| t != "object") {
        return Err("tool schema branch type must be object".into());
    }
    if let Some((_, branches)) = combinator(schema)? {
        for branch in branches {
            if let Some(obj) = branch.as_object() {
                validate_branch_tree(obj)?;
            } else if branch != &Value::Bool(true) {
                return Err("tool schema branch must be an object or true".into());
            }
        }
    }
    Ok(())
}

fn required(obj: &Map<String, Value>) -> BTreeSet<String> {
    obj.get("required")
        .and_then(Value::as_array)
        .into_iter()
        .flatten()
        .filter_map(Value::as_str)
        .map(str::to_owned)
        .collect()
}

struct Flattened {
    properties: BTreeMap<String, Value>,
    required: BTreeSet<String>,
}

fn flatten(obj: &Map<String, Value>) -> Flattened {
    let mut props: BTreeMap<String, Value> = obj
        .get("properties")
        .and_then(Value::as_object)
        .into_iter()
        .flat_map(|p| p.iter())
        .map(|(k, v)| (k.clone(), v.clone()))
        .collect();
    let mut must = required(obj);
    if let Ok(Some((kind, branches))) = combinator(obj) {
        let children: Vec<Flattened> = branches
            .iter()
            .map(|v| {
                v.as_object().map(flatten).unwrap_or(Flattened {
                    properties: BTreeMap::new(),
                    required: BTreeSet::new(),
                })
            })
            .collect();
        let mut branch_required: Option<BTreeSet<String>> = None;
        let mut variants: BTreeMap<String, Vec<Value>> = BTreeMap::new();
        for child in &children {
            branch_required = Some(match branch_required {
                None => child.required.clone(),
                Some(prev) if kind == "allOf" => prev.union(&child.required).cloned().collect(),
                Some(prev) => prev.intersection(&child.required).cloned().collect(),
            });
            for (key, schema) in &child.properties {
                let entry = variants.entry(key.clone()).or_default();
                if !entry.contains(schema) {
                    entry.push(schema.clone());
                }
            }
        }
        must.extend(branch_required.unwrap_or_default());
        for (key, schemas) in variants {
            let merged = if schemas.len() == 1 {
                schemas.into_iter().next().unwrap()
            } else if kind == "allOf" {
                json!({"allOf": schemas})
            } else {
                json!({"anyOf": schemas})
            };
            if let Some(root) = props.get(&key) {
                if root != &merged {
                    props.insert(key, json!({"allOf": [root, merged]}));
                }
            } else {
                props.insert(key, merged);
            }
        }
    }
    Flattened {
        properties: props,
        required: must,
    }
}

/// Build a Claude or Grok tool schema. Invalid provider-root shapes fail before dispatch.
fn grok_nested_unions(value: &mut Value) {
    let Value::Object(obj) = value else {
        return;
    };
    if let Some(one_of) = obj.remove("oneOf") {
        if obj.contains_key("anyOf") {
            obj.insert("oneOf".into(), one_of);
        } else {
            obj.insert("anyOf".into(), one_of);
        }
    }
    for key in [
        "properties",
        "patternProperties",
        "$defs",
        "definitions",
        "dependentSchemas",
    ] {
        if let Some(entries) = obj.get_mut(key).and_then(Value::as_object_mut) {
            for child in entries.values_mut() {
                grok_nested_unions(child);
            }
        }
    }
    for key in [
        "items",
        "additionalProperties",
        "additionalItems",
        "unevaluatedProperties",
        "unevaluatedItems",
        "not",
        "if",
        "then",
        "else",
        "contains",
        "propertyNames",
    ] {
        if let Some(child) = obj.get_mut(key) {
            grok_nested_unions(child);
        }
    }
    for key in ["prefixItems", "oneOf", "anyOf", "allOf"] {
        if let Some(values) = obj.get_mut(key).and_then(Value::as_array_mut) {
            for child in values {
                grok_nested_unions(child);
            }
        }
    }
}

pub fn provider_tool_schema(schema: &Value, grok: bool) -> Result<Value, String> {
    let obj = schema
        .as_object()
        .ok_or("tool schema root must be an object")?;
    validate_branch_tree(obj)?;
    let Some((kind, branches)) = combinator(obj)? else {
        let mut copy = obj.clone();
        copy.entry("type").or_insert(json!("object"));
        copy.entry("properties").or_insert(json!({}));
        let mut result = Value::Object(copy);
        if grok {
            grok_nested_unions(&mut result);
        }
        return Ok(result);
    };
    if grok && kind != "allOf" && branches.iter().all(|v| v.is_object()) {
        let mut copy = obj.clone();
        let mut normalized = branches.to_vec();
        for branch in &mut normalized {
            branch
                .as_object_mut()
                .unwrap()
                .entry("type")
                .or_insert(json!("object"));
        }
        for branch in &mut normalized {
            grok_nested_unions(branch);
        }
        copy.insert(kind.into(), Value::Array(normalized));
        copy.entry("type").or_insert(json!("object"));
        copy.entry("properties").or_insert(json!({}));
        let root_union = copy.remove(kind).unwrap();
        let mut root = Value::Object(copy);
        grok_nested_unions(&mut root);
        let mut copy = root.as_object().unwrap().clone();
        copy.insert(kind.into(), root_union);
        return Ok(Value::Object(copy));
    }
    let flat = flatten(obj);
    let mut copy = obj.clone();
    for key in COMBINATORS {
        copy.remove(key);
    }
    copy.insert("type".into(), json!("object"));
    copy.insert("properties".into(), json!(flat.properties));
    if !flat.required.is_empty() {
        copy.insert("required".into(), json!(flat.required));
    }
    let names: Vec<&str> = flat.properties.keys().map(String::as_str).collect();
    let sentence = format!(
        "Flattened {kind} tool schema with properties: {}.",
        names.join(", ")
    );
    let description = copy
        .get("description")
        .and_then(Value::as_str)
        .unwrap_or("");
    copy.insert(
        "description".into(),
        json!(if description.is_empty() {
            sentence
        } else {
            format!("{description} {sentence}")
        }),
    );
    let mut result = Value::Object(copy);
    if grok {
        grok_nested_unions(&mut result);
    }
    Ok(result)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn flatten_preserves_root_and_nested_required() {
        let schema = json!({"type":"object", "description":"Original", "additionalProperties":false,
            "properties":{"b":{"type":"string"}}, "required":["b"],
            "oneOf":[{"required":["a"], "anyOf":[{"properties":{"a":{"type":"number"}},"required":["x"]},
                {"properties":{"c":{"type":"boolean"}},"required":["y"]}]},
                {"required":["c"],"properties":{"d":{"type":"number"}}}]});
        let flat = provider_tool_schema(&schema, false).unwrap();
        assert_eq!(flat["type"], "object");
        assert_eq!(flat["additionalProperties"], false);
        assert_eq!(flat["required"], json!(["b"]));
        for key in ["a", "b", "c", "d"] {
            assert!(flat["properties"].get(key).is_some());
        }
        assert!(flat.get("oneOf").is_none());
        assert!(flat["description"].as_str().unwrap().contains("oneOf"));
    }

    #[test]
    fn flatten_collisions_and_conjunction() {
        let schema = json!({"properties":{"x":{"type":"string"}}, "oneOf":[
            {"properties":{"x":{"type":"number"}}}, {"properties":{"x":{"type":"boolean"}}}]});
        let x = &provider_tool_schema(&schema, false).unwrap()["properties"]["x"];
        assert_eq!(
            x,
            &json!({"allOf":[{"type":"string"}, {"anyOf":[{"type":"number"},{"type":"boolean"}]}]})
        );
        let all = json!({"properties":{"a":{"type":"string"}},"allOf":[{"properties":{"b":{"type":"number"}},"required":["b"]}]});
        let flat = provider_tool_schema(&all, false).unwrap();
        assert_eq!(flat["properties"]["a"], json!({"type":"string"}));
        assert_eq!(flat["properties"]["b"], json!({"type":"number"}));
        assert_eq!(flat["required"], json!(["b"]));
        let same = json!({"allOf":[{"properties":{"x":{"type":"string"}}},{"properties":{"x":{"type":"number"}}}]});
        assert_eq!(
            provider_tool_schema(&same, false).unwrap()["properties"]["x"],
            json!({"allOf":[{"type":"string"},{"type":"number"}]})
        );
    }

    #[test]
    fn grok_copy_and_true_flatten() {
        let schema = json!({"type":"object","oneOf":[{"properties":{"x":{"type":"string"}}},{"type":"object"}]});
        let grok = provider_tool_schema(&schema, true).unwrap();
        assert_eq!(grok["oneOf"][0]["type"], "object");
        assert!(
            provider_tool_schema(&schema, false)
                .unwrap()
                .get("oneOf")
                .is_none()
        );
        for grok in [true, false] {
            let flat = provider_tool_schema(&json!({"anyOf":[true]}), grok).unwrap();
            assert_eq!(flat["properties"], json!({}));
            assert!(flat.get("anyOf").is_none());
        }
    }

    #[test]
    fn strict_protocol_rules_and_downgrades() {
        let schema = json!({"type":"object", "additionalProperties":false, "properties":{
            "x":{"anyOf":[{"type":"object","additionalProperties":false,"properties":{"y":{"type":"string"}},"required":["y"]},
                {"type":"null"}] }}, "required":["x"]});
        validate_tool_schema(&schema, true, false).unwrap();
        assert!(claude_strict(&schema, true));
        assert!(codex_strict(&schema, true));
        let mut loose = schema.clone();
        loose["required"] = json!([]);
        assert!(claude_strict(&loose, true));
        assert!(!codex_strict(&loose, true));
        loose["additionalProperties"] = json!(true);
        assert!(!claude_strict(&loose, true));
        assert!(!codex_strict(&loose, true));
        let all = json!({"type":"object","properties":{"x":{"allOf":[{"type":"string"}]}}});
        assert!(validate_tool_schema(&all, true, false).is_err());
        validate_tool_schema(&all, true, true).unwrap();
        assert!(
            validate_tool_schema(
                &json!({"type":"object","properties":{"x":{"allOf":[{"$ref":"#/$defs/x"}]}}}),
                true,
                true
            )
            .is_err()
        );
    }

    #[test]
    fn reject_invalid_combinators_and_provider_branches() {
        for bad in [
            json!({"oneOf":[]}),
            json!({"properties":{"x":{"anyOf":null}}}),
        ] {
            assert!(validate_tool_schema(&bad, false, false).is_err());
            assert!(validate_tool_schema(&bad, false, true).is_err());
        }
        assert!(validate_tool_schema(&json!({"anyOf":[{"type":"object"}]}), false, true).is_err());
        assert!(
            validate_tool_schema(
                &json!({"type":"object","oneOf":[{"type":"object"}]}),
                true,
                false
            )
            .is_err()
        );
        for bad in [
            json!({"type":"string"}),
            json!({"oneOf":[{}],"allOf":[{}]}),
            json!({"anyOf":[{}, {"type":"string"}]}),
            json!({"anyOf":[{},7]}),
            json!({"oneOf":[{"anyOf":[{"type":"number"}]}]}),
        ] {
            assert!(provider_tool_schema(&bad, false).is_err());
            assert!(provider_tool_schema(&bad, true).is_err());
        }
    }
    #[test]
    fn keyword_property_names_are_data_not_schema_keywords() {
        let schema = json!({"type":"object","properties":{"anyOf":{"type":"string"},"oneOf":{"type":"string","default":{"oneOf":7}}}});
        validate_tool_schema(&schema, false, false).unwrap();
        let copy = provider_tool_schema(&schema, true).unwrap();
        assert_eq!(copy["properties"]["oneOf"]["default"]["oneOf"], 7);
        assert!(copy["properties"].get("anyOf").is_some());
    }

    #[test]
    fn true_branch_does_not_require_other_branches() {
        let schema =
            json!({"anyOf":[true,{"properties":{"x":{"type":"string"}},"required":["x"]}]});
        for grok in [false, true] {
            let flat = provider_tool_schema(&schema, grok).unwrap();
            assert!(flat.get("required").is_none());
        }
    }

    #[test]
    fn nested_keywords_and_collisions_are_not_dropped() {
        let nested = json!({"type":"object","properties":{"x":{"oneOf":[{"type":"string"}],"anyOf":[{"type":"number"}]}}});
        validate_tool_schema(&nested, false, false).unwrap();
        let grok = provider_tool_schema(&nested, true).unwrap();
        assert!(grok["properties"]["x"]["anyOf"].is_array());
        assert!(grok["properties"]["x"]["oneOf"].is_array());
        assert!(
            validate_tool_schema(&json!({"properties":{"x":{"anyOf":[{},7]}}}), false, false)
                .is_ok()
        );
        assert!(
            provider_tool_schema(
                &json!({"anyOf":[{"properties":{"x":{"anyOf":[{},7]}}}]}),
                false
            )
            .is_ok()
        );
        let colliding = json!({"allOf":[{"anyOf":[{"properties":{"x":{"minimum":0}}}]},{"properties":{"x":{"maximum":10}}}]});
        assert!(
            provider_tool_schema(&colliding, false).unwrap()["properties"]["x"]["allOf"].is_array()
        );
    }

    #[test]
    fn nullable_object_never_gets_strict_true() {
        let schema = json!({"type":"object","additionalProperties":false,"required":["x"],"properties":{"x":{"type":["object","null"]}}});
        assert!(!claude_strict(&schema, true));
        assert!(!codex_strict(&schema, true));
    }
    #[test]
    fn grok_root_keep_preserves_keyword_property_and_maps_nested_union() {
        let schema = json!({"type":"object","oneOf":[{"type":"object"}],"properties":{
            "oneOf":{"type":"string"},"x":{"oneOf":[{"type":"string"}]}}});
        let copy = provider_tool_schema(&schema, true).unwrap();
        assert!(copy["properties"].get("oneOf").is_some());
        assert_eq!(copy["properties"]["x"]["anyOf"], json!([{"type":"string"}]));
        assert!(copy["properties"]["x"].get("oneOf").is_none());
        assert!(copy.get("oneOf").is_some());
    }

    #[test]
    fn dependent_and_tuple_subschemas_are_validated() {
        for schema in [
            json!({"dependentSchemas":{"a":{"anyOf":[]}}}),
            json!({"items":[{"oneOf":[]}]}),
            json!({"unevaluatedProperties":{"allOf":[]}}),
        ] {
            assert!(validate_tool_schema(&schema, false, false).is_err());
        }
        let schema = json!({"type":"object","additionalProperties":false,"properties":{},
            "dependentSchemas":{"a":{"oneOf":[{"type":"string"}]}}});
        assert!(validate_tool_schema(&schema, true, false).is_err());
        assert!(!claude_strict(&schema, true));
        assert!(!codex_strict(&schema, true));
    }
}
