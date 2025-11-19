"""
Knowledge Base Query Interface

High-level interface for querying the SAP Knowledge Base.
"""

import re
from datetime import datetime
from typing import Any, Dict, List, Optional

from sap_llm.knowledge_base.field_mappings import FieldMappingManager
from sap_llm.knowledge_base.storage import KnowledgeBaseStorage
from sap_llm.utils.logger import get_logger

logger = get_logger(__name__)


class KnowledgeBaseQuery:
    """
    High-level interface for querying SAP API knowledge.

    Provides convenient methods for:
    - Finding APIs for document types
    - Mapping ADC fields to SAP fields
    - Finding relevant business rules
    - Generating transformation code
    """

    def __init__(self, storage: KnowledgeBaseStorage):
        """
        Initialize query interface.

        Args:
            storage: Knowledge base storage instance
        """
        self.storage = storage
        self.field_mapping_manager = FieldMappingManager()
        logger.info("Knowledge Base Query initialized")

    def find_api_for_document(
        self, doc_type: str, doc_subtype: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find SAP APIs for document type.

        Args:
            doc_type: Document type (e.g., "purchase_order")
            doc_subtype: Document subtype (e.g., "standard")

        Returns:
            List of relevant API schemas
        """
        # Build query
        query_parts = [doc_type.replace("_", " ")]
        if doc_subtype:
            query_parts.append(doc_subtype)

        query = " ".join(query_parts)

        logger.debug(f"Finding APIs for: {query}")

        # Search APIs
        apis = self.storage.search_apis(query, k=5)

        logger.info(f"Found {len(apis)} APIs for {doc_type}")

        return apis

    def map_fields_to_sap(
        self, adc_data: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """
        Map ADC fields to SAP API fields.

        Args:
            adc_data: ADC dictionary
            doc_type: Document type

        Returns:
            Dictionary with SAP field mappings
        """
        sap_payload = {}
        field_mappings = {}

        for field_name, field_value in adc_data.items():
            # Search for field mapping
            mappings = self.storage.search_field_mappings(field_name, k=1)

            if mappings:
                mapping = mappings[0]
                sap_field = mapping.get("sap_field", field_name)
                field_mappings[field_name] = sap_field

                # Apply transformation if available
                transformed_value = self._apply_field_transformation(
                    field_value, mapping
                )

                sap_payload[sap_field] = transformed_value

            else:
                # No mapping found, use original name
                logger.warning(f"No mapping found for field: {field_name}")
                sap_payload[field_name] = field_value

        return {
            "payload": sap_payload,
            "mappings": field_mappings,
        }

    def _apply_field_transformation(
        self, value: Any, mapping: Dict[str, Any]
    ) -> Any:
        """
        Apply field transformation based on mapping.

        Args:
            value: Original value
            mapping: Field mapping with transformation rules

        Returns:
            Transformed value
        """
        # Check for transformation function
        transform_type = mapping.get("transform")

        if transform_type == "uppercase":
            return str(value).upper()
        elif transform_type == "lowercase":
            return str(value).lower()
        elif transform_type == "remove_spaces":
            return str(value).replace(" ", "")
        elif transform_type == "date_format":
            # Implement date formatting for SAP
            return self._format_date(value, mapping.get("target_format", "SAP"))
        else:
            return value

    def _format_date(self, value: Any, target_format: str = "SAP") -> str:
        """
        Parse and format date values for SAP systems.

        Args:
            value: Date value (string, datetime, or timestamp)
            target_format: Target format (SAP, ISO, or custom format string)

        Returns:
            Formatted date string
        """
        try:
            # If already a datetime object
            if isinstance(value, datetime):
                date_obj = value
            else:
                # Parse string date in common formats
                date_str = str(value).strip()

                # Try common date formats
                date_formats = [
                    "%Y-%m-%d",           # ISO format: 2024-01-15
                    "%d/%m/%Y",           # European: 15/01/2024
                    "%m/%d/%Y",           # US: 01/15/2024
                    "%Y%m%d",             # SAP format: 20240115
                    "%d.%m.%Y",           # German: 15.01.2024
                    "%Y-%m-%dT%H:%M:%S",  # ISO with time
                    "%Y-%m-%d %H:%M:%S",  # SQL timestamp
                ]

                date_obj = None
                for fmt in date_formats:
                    try:
                        date_obj = datetime.strptime(date_str, fmt)
                        break
                    except ValueError:
                        continue

                if date_obj is None:
                    logger.warning(f"Could not parse date: {value}")
                    return str(value)

            # Format based on target
            if target_format == "SAP":
                # SAP standard format: YYYYMMDD
                return date_obj.strftime("%Y%m%d")
            elif target_format == "ISO":
                # ISO 8601 format
                return date_obj.strftime("%Y-%m-%d")
            else:
                # Custom format string
                return date_obj.strftime(target_format)

        except Exception as e:
            logger.error(f"Error formatting date {value}: {e}")
            return str(value)

    def find_validation_rules(
        self, doc_type: str, field_name: Optional[str] = None
    ) -> List[Dict[str, Any]]:
        """
        Find validation rules for document or field.

        Args:
            doc_type: Document type
            field_name: Optional specific field name

        Returns:
            List of applicable business rules
        """
        # Build query
        query_parts = [doc_type.replace("_", " "), "validation"]
        if field_name:
            query_parts.append(field_name)

        query = " ".join(query_parts)

        logger.debug(f"Finding validation rules for: {query}")

        # Search rules
        rules = self.storage.search_business_rules(query, k=10)

        # Filter by type
        validation_rules = [r for r in rules if r.get("type") == "validation"]

        logger.info(f"Found {len(validation_rules)} validation rules")

        return validation_rules

    def find_calculation_rules(
        self, doc_type: str
    ) -> List[Dict[str, Any]]:
        """
        Find calculation rules for document type.

        Args:
            doc_type: Document type

        Returns:
            List of calculation rules
        """
        query = f"{doc_type.replace('_', ' ')} calculation formula"

        logger.debug(f"Finding calculation rules for: {query}")

        # Search rules
        rules = self.storage.search_business_rules(query, k=10)

        # Filter by type
        calc_rules = [r for r in rules if r.get("type") == "calculation"]

        logger.info(f"Found {len(calc_rules)} calculation rules")

        return calc_rules

    def get_endpoint_for_action(
        self, doc_type: str, action: str = "create"
    ) -> Optional[Dict[str, Any]]:
        """
        Get API endpoint for specific action.

        Args:
            doc_type: Document type
            action: Action (create, update, delete)

        Returns:
            Endpoint information or None
        """
        # Find APIs
        apis = self.find_api_for_document(doc_type)

        if not apis:
            logger.warning(f"No APIs found for {doc_type}")
            return None

        # Get best matching API
        api = apis[0]

        # Find endpoint for action
        endpoints = api.get("endpoints", [])

        # Map action to HTTP method
        method_map = {
            "create": "POST",
            "update": ["PUT", "PATCH"],
            "delete": "DELETE",
            "read": "GET",
        }

        target_methods = method_map.get(action, "POST")
        if not isinstance(target_methods, list):
            target_methods = [target_methods]

        # Find matching endpoint
        for endpoint in endpoints:
            if endpoint.get("method") in target_methods:
                return {
                    "api": api.get("title"),
                    "endpoint": endpoint.get("path"),
                    "method": endpoint.get("method"),
                    "description": endpoint.get("description", ""),
                }

        logger.warning(f"No endpoint found for {action} on {doc_type}")
        return None

    def get_example_payload(
        self, doc_type: str, action: str = "create"
    ) -> Optional[Dict[str, Any]]:
        """
        Get example payload for document type.

        Args:
            doc_type: Document type
            action: Action type

        Returns:
            Example payload or None
        """
        logger.debug(f"Getting example payload for {doc_type} {action}")

        try:
            # Query examples from storage
            if not self.storage.mock_mode and self.storage.examples_col:
                # Search MongoDB for matching examples
                query = {
                    "doc_type": doc_type,
                    "action": action
                }

                # Try exact match first
                example = self.storage.examples_col.find_one(query)

                if example:
                    # Remove MongoDB _id field
                    example.pop("_id", None)
                    logger.info(f"Found example for {doc_type} {action}")
                    return example

                # Try without action if no exact match
                example = self.storage.examples_col.find_one({"doc_type": doc_type})

                if example:
                    example.pop("_id", None)
                    logger.info(f"Found generic example for {doc_type}")
                    return example

            # Fallback: Generate example from API schema
            apis = self.find_api_for_document(doc_type)

            if apis:
                api = apis[0]
                endpoints = api.get("endpoints", [])

                # Find endpoint matching action
                method_map = {
                    "create": "POST",
                    "update": ["PUT", "PATCH"],
                    "delete": "DELETE",
                    "read": "GET",
                }

                target_methods = method_map.get(action, "POST")
                if not isinstance(target_methods, list):
                    target_methods = [target_methods]

                for endpoint in endpoints:
                    if endpoint.get("method") in target_methods:
                        # Extract example from request body or schema
                        request_body = endpoint.get("request_body", {})
                        example_payload = self._extract_example_from_schema(request_body)

                        if example_payload:
                            logger.info(f"Generated example from API schema for {doc_type}")
                            return {
                                "doc_type": doc_type,
                                "action": action,
                                "payload": example_payload,
                                "api": api.get("title"),
                                "endpoint": endpoint.get("path"),
                            }

            logger.warning(f"No example found for {doc_type} {action}")
            return None

        except Exception as e:
            logger.error(f"Error retrieving example payload: {e}")
            return None

    def _extract_example_from_schema(self, schema: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        """
        Extract example payload from OpenAPI schema.

        Args:
            schema: OpenAPI schema or request body definition

        Returns:
            Example payload or None
        """
        try:
            # Check for explicit example
            if "example" in schema:
                return schema["example"]

            # Check for examples array
            if "examples" in schema and isinstance(schema["examples"], dict):
                # Return first example
                for example_value in schema["examples"].values():
                    if isinstance(example_value, dict) and "value" in example_value:
                        return example_value["value"]
                    return example_value

            # Extract from content schema
            content = schema.get("content", {})
            for content_type, content_schema in content.items():
                if "example" in content_schema:
                    return content_schema["example"]

                # Check schema properties
                schema_def = content_schema.get("schema", {})
                if "example" in schema_def:
                    return schema_def["example"]

                # Generate example from properties
                properties = schema_def.get("properties", {})
                if properties:
                    example = {}
                    for prop_name, prop_def in properties.items():
                        if "example" in prop_def:
                            example[prop_name] = prop_def["example"]
                        elif "default" in prop_def:
                            example[prop_name] = prop_def["default"]
                        else:
                            # Generate placeholder based on type
                            prop_type = prop_def.get("type", "string")
                            example[prop_name] = self._generate_example_value(prop_type)

                    return example

            return None

        except Exception as e:
            logger.error(f"Error extracting example from schema: {e}")
            return None

    def _generate_example_value(self, field_type: str) -> Any:
        """
        Generate example value for field type.

        Args:
            field_type: Field type (string, number, boolean, etc.)

        Returns:
            Example value
        """
        type_examples = {
            "string": "example_value",
            "number": 0,
            "integer": 0,
            "boolean": False,
            "array": [],
            "object": {},
        }

        return type_examples.get(field_type, "example_value")

    def validate_payload(
        self, payload: Dict[str, Any], doc_type: str
    ) -> Dict[str, Any]:
        """
        Validate payload against business rules.

        Args:
            payload: Payload to validate
            doc_type: Document type

        Returns:
            Validation result with errors
        """
        errors = []
        warnings = []

        # Get validation rules
        rules = self.find_validation_rules(doc_type)

        for rule in rules:
            # Apply rule
            result = self._apply_validation_rule(payload, rule)

            if not result["valid"]:
                errors.append({
                    "rule_id": rule.get("rule_id"),
                    "description": rule.get("description"),
                    "details": result.get("details"),
                })

        return {
            "valid": len(errors) == 0,
            "errors": errors,
            "warnings": warnings,
        }

    def _apply_validation_rule(
        self, payload: Dict[str, Any], rule: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Apply validation rule to payload.

        Args:
            payload: Payload to validate
            rule: Validation rule

        Returns:
            Validation result
        """
        try:
            rule_type = rule.get("type", "general")

            # Pattern validation (regex)
            if rule_type == "validation" and "pattern" in rule:
                field_name = self._extract_field_from_rule(rule.get("description", ""))

                if field_name and field_name in payload:
                    value = str(payload[field_name])
                    pattern = rule["pattern"]

                    if not re.match(pattern, value):
                        return {
                            "valid": False,
                            "details": f"Field '{field_name}' does not match pattern {pattern}",
                        }

            # Required field validation
            elif rule_type == "required":
                field_name = self._extract_field_from_rule(rule.get("description", ""))

                if field_name and field_name not in payload:
                    return {
                        "valid": False,
                        "details": f"Required field '{field_name}' is missing",
                    }

            # Range validation
            elif rule_type == "range":
                field_name = self._extract_field_from_rule(rule.get("description", ""))

                if field_name and field_name in payload:
                    value = payload[field_name]

                    # Extract range from rule
                    range_match = re.search(r"between\s+(\d+(?:\.\d+)?)\s+and\s+(\d+(?:\.\d+)?)", rule.get("description", ""), re.IGNORECASE)

                    if range_match:
                        min_val = float(range_match.group(1))
                        max_val = float(range_match.group(2))

                        try:
                            num_value = float(value)
                            if not (min_val <= num_value <= max_val):
                                return {
                                    "valid": False,
                                    "details": f"Field '{field_name}' value {num_value} not in range [{min_val}, {max_val}]",
                                }
                        except (ValueError, TypeError):
                            return {
                                "valid": False,
                                "details": f"Field '{field_name}' is not a valid number",
                            }

            # Formula/Calculation validation
            elif rule_type == "calculation" and "formula" in rule:
                formula = rule["formula"]
                result = self._evaluate_formula(formula, payload)

                if not result["valid"]:
                    return {
                        "valid": False,
                        "details": result.get("error", "Formula validation failed"),
                    }

            # Format validation
            elif rule_type == "format":
                field_name = self._extract_field_from_rule(rule.get("description", ""))

                if field_name and field_name in payload:
                    # Check specific format requirements
                    description = rule.get("description", "").lower()

                    if "email" in description:
                        email_pattern = r"^[a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,}$"
                        if not re.match(email_pattern, str(payload[field_name])):
                            return {
                                "valid": False,
                                "details": f"Field '{field_name}' is not a valid email",
                            }

                    elif "phone" in description:
                        # Basic phone validation
                        phone_pattern = r"^\+?[\d\s\-\(\)]+$"
                        if not re.match(phone_pattern, str(payload[field_name])):
                            return {
                                "valid": False,
                                "details": f"Field '{field_name}' is not a valid phone number",
                            }

            # Rule passed or not applicable
            return {"valid": True}

        except Exception as e:
            logger.error(f"Error applying validation rule: {e}")
            return {
                "valid": False,
                "details": f"Error applying rule: {str(e)}",
            }

    def _extract_field_from_rule(self, description: str) -> Optional[str]:
        """
        Extract field name from rule description.

        Args:
            description: Rule description

        Returns:
            Field name or None
        """
        # Try to extract field name from common patterns
        patterns = [
            r"field\s+'(\w+)'",          # field 'fieldname'
            r"field\s+\"(\w+)\"",        # field "fieldname"
            r"'(\w+)'\s+field",          # 'fieldname' field
            r"^(\w+)\s+must",            # fieldname must
            r"^(\w+)\s+should",          # fieldname should
        ]

        for pattern in patterns:
            match = re.search(pattern, description, re.IGNORECASE)
            if match:
                return match.group(1)

        return None

    def _evaluate_formula(self, formula: str, payload: Dict[str, Any]) -> Dict[str, Any]:
        """
        Evaluate calculation formula against payload.

        Args:
            formula: Formula string (e.g., "total_amount = subtotal + tax_amount")
            payload: Payload data

        Returns:
            Validation result
        """
        try:
            # Parse formula: expected_field = expression
            formula_parts = formula.split("=")

            if len(formula_parts) != 2:
                return {
                    "valid": False,
                    "error": "Invalid formula format",
                }

            target_field = formula_parts[0].strip()
            expression = formula_parts[1].strip()

            # Check if target field exists in payload
            if target_field not in payload:
                return {
                    "valid": True,  # Field not present, rule doesn't apply
                }

            # Extract field names from expression
            field_pattern = r"\b([a-zA-Z_]\w*)\b"
            fields_in_expr = re.findall(field_pattern, expression)

            # Check if all required fields are present
            missing_fields = [f for f in fields_in_expr if f not in payload and not f.isdigit()]

            if missing_fields:
                return {
                    "valid": False,
                    "error": f"Missing fields for calculation: {', '.join(missing_fields)}",
                }

            # Build safe evaluation context
            eval_context = {}
            for field in fields_in_expr:
                if field in payload:
                    try:
                        eval_context[field] = float(payload[field])
                    except (ValueError, TypeError):
                        eval_context[field] = payload[field]

            # Evaluate expression safely (only allow basic arithmetic)
            try:
                # SECURITY: Never use eval() - it's vulnerable to code injection
                # Use ast module for safe evaluation
                import ast
                import operator

                # Define allowed operations
                allowed_operators = {
                    ast.Add: operator.add,
                    ast.Sub: operator.sub,
                    ast.Mult: operator.mul,
                    ast.Div: operator.truediv,
                    ast.Mod: operator.mod,
                    ast.Pow: operator.pow,
                    ast.USub: operator.neg,
                    ast.UAdd: operator.pos,
                }

                def safe_eval(node, context):
                    """Safely evaluate arithmetic expression using AST."""
                    if isinstance(node, ast.BinOp):
                        op_func = allowed_operators.get(type(node.op))
                        if not op_func:
                            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
                        left = safe_eval(node.left, context)
                        right = safe_eval(node.right, context)
                        return op_func(left, right)
                    elif isinstance(node, ast.UnaryOp):
                        op_func = allowed_operators.get(type(node.op))
                        if not op_func:
                            raise ValueError(f"Operator {type(node.op).__name__} not allowed")
                        operand = safe_eval(node.operand, context)
                        return op_func(operand)
                    elif isinstance(node, ast.Num):  # Python < 3.8
                        return node.n
                    elif isinstance(node, ast.Constant):  # Python >= 3.8
                        return node.value
                    elif isinstance(node, ast.Name):
                        if node.id in context:
                            return context[node.id]
                        raise ValueError(f"Variable '{node.id}' not found in context")
                    else:
                        raise ValueError(f"Expression type {type(node).__name__} not allowed")

                # Parse and evaluate expression
                parsed_expr = ast.parse(expression, mode='eval')
                calculated_value = safe_eval(parsed_expr.body, eval_context)
                expected_value = float(payload[target_field])

                # Compare with tolerance for floating point
                tolerance = 0.01
                if abs(calculated_value - expected_value) > tolerance:
                    return {
                        "valid": False,
                        "error": f"Calculation mismatch: expected {calculated_value}, got {expected_value}",
                    }

                return {"valid": True}

            except Exception as e:
                return {
                    "valid": False,
                    "error": f"Error evaluating formula: {str(e)}",
                }

        except Exception as e:
            logger.error(f"Error evaluating formula: {e}")
            return {
                "valid": False,
                "error": str(e),
            }

    def get_transformation_code(
        self, source_format: str, target_format: str
    ) -> Optional[str]:
        """
        Get transformation code between formats.

        Args:
            source_format: Source format (e.g., "ADC")
            target_format: Target format (e.g., "SAP_ODATA")

        Returns:
            Python transformation code or None
        """
        logger.debug(f"Getting transformation: {source_format} -> {target_format}")

        try:
            # Generate transformation code based on format types
            if source_format.upper() == "ADC" and target_format.upper() in ["SAP_ODATA", "SAP"]:
                return self._generate_adc_to_sap_code()

            elif source_format.upper() == "SAP" and target_format.upper() == "ADC":
                return self._generate_sap_to_adc_code()

            elif source_format.upper() == "JSON" and target_format.upper() == "SAP_ODATA":
                return self._generate_json_to_odata_code()

            else:
                # Generic transformation template
                return self._generate_generic_transformation_code(source_format, target_format)

        except Exception as e:
            logger.error(f"Error generating transformation code: {e}")
            return None

    def _generate_adc_to_sap_code(self) -> str:
        """
        Generate transformation code from ADC to SAP format.

        Returns:
            Python transformation code
        """
        code = '''def transform_adc_to_sap(adc_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform ADC format to SAP OData format.

    Args:
        adc_data: ADC dictionary

    Returns:
        SAP-formatted payload
    """
    from datetime import datetime

    sap_payload = {}

    # Field mappings (customize based on document type)
    field_map = {
        "po_number": "PurchaseOrderID",
        "vendor_id": "VendorID",
        "vendor_name": "VendorName",
        "po_date": "PurchaseOrderDate",
        "total_amount": "TotalAmount",
        "currency": "Currency",
        "company_code": "CompanyCode",
    }

    # Apply field mappings
    for adc_field, sap_field in field_map.items():
        if adc_field in adc_data:
            value = adc_data[adc_field]

            # Apply transformations
            if "date" in adc_field.lower():
                # Convert to SAP date format (YYYYMMDD)
                try:
                    if isinstance(value, str):
                        date_obj = datetime.strptime(value, "%Y-%m-%d")
                        value = date_obj.strftime("%Y%m%d")
                except:
                    pass

            sap_payload[sap_field] = value

    # Add line items if present
    if "line_items" in adc_data:
        sap_payload["PurchaseOrderItems"] = []

        for idx, item in enumerate(adc_data["line_items"]):
            sap_item = {
                "ItemNumber": str(idx + 1).zfill(5),
                "MaterialID": item.get("material_id", ""),
                "Quantity": item.get("quantity", 0),
                "UnitPrice": item.get("unit_price", 0),
                "TotalPrice": item.get("total_price", 0),
            }
            sap_payload["PurchaseOrderItems"].append(sap_item)

    return sap_payload
'''
        return code

    def _generate_sap_to_adc_code(self) -> str:
        """
        Generate transformation code from SAP to ADC format.

        Returns:
            Python transformation code
        """
        code = '''def transform_sap_to_adc(sap_data: Dict[str, Any]) -> Dict[str, Any]:
    """
    Transform SAP format to ADC format.

    Args:
        sap_data: SAP OData response

    Returns:
        ADC-formatted dictionary
    """
    from datetime import datetime

    adc_data = {}

    # Reverse field mappings
    field_map = {
        "PurchaseOrderID": "po_number",
        "VendorID": "vendor_id",
        "VendorName": "vendor_name",
        "PurchaseOrderDate": "po_date",
        "TotalAmount": "total_amount",
        "Currency": "currency",
        "CompanyCode": "company_code",
    }

    # Apply field mappings
    for sap_field, adc_field in field_map.items():
        if sap_field in sap_data:
            value = sap_data[sap_field]

            # Apply transformations
            if "date" in adc_field.lower():
                # Convert from SAP date format to ISO
                try:
                    if isinstance(value, str) and len(value) == 8:
                        date_obj = datetime.strptime(value, "%Y%m%d")
                        value = date_obj.strftime("%Y-%m-%d")
                except:
                    pass

            adc_data[adc_field] = value

    # Transform line items
    if "PurchaseOrderItems" in sap_data:
        adc_data["line_items"] = []

        for item in sap_data["PurchaseOrderItems"]:
            adc_item = {
                "material_id": item.get("MaterialID", ""),
                "quantity": item.get("Quantity", 0),
                "unit_price": item.get("UnitPrice", 0),
                "total_price": item.get("TotalPrice", 0),
            }
            adc_data["line_items"].append(adc_item)

    return adc_data
'''
        return code

    def _generate_json_to_odata_code(self) -> str:
        """
        Generate transformation code from JSON to OData format.

        Returns:
            Python transformation code
        """
        code = '''def transform_json_to_odata(json_data: Dict[str, Any]) -> str:
    """
    Transform JSON to OData query format.

    Args:
        json_data: JSON dictionary

    Returns:
        OData query string
    """
    import json
    from urllib.parse import quote

    # Convert to OData JSON format
    odata_payload = json.dumps(json_data)

    # OData headers
    headers = {
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    return odata_payload
'''
        return code

    def _generate_generic_transformation_code(
        self, source_format: str, target_format: str
    ) -> str:
        """
        Generate generic transformation code template.

        Args:
            source_format: Source format
            target_format: Target format

        Returns:
            Python transformation code template
        """
        code = f'''def transform_{source_format.lower()}_to_{target_format.lower()}(
    source_data: Dict[str, Any]
) -> Dict[str, Any]:
    """
    Transform {source_format} format to {target_format} format.

    Args:
        source_data: Source data dictionary

    Returns:
        Transformed data dictionary
    """
    target_data = {{}}

    # Field mappings loaded from knowledge base
    field_map = {{
        # Common SAP field mappings
        "invoice_number": "BELNR",
        "invoice_date": "BLDAT",
        "posting_date": "BUDAT",
        "vendor_id": "LIFNR",
        "vendor_name": "NAME1",
        "total_amount": "WRBTR",
        "currency": "WAERS",
        "tax_amount": "MWSTS",
        "payment_terms": "ZTERM",
        "purchase_order": "EBELN",
        "company_code": "BUKRS",
        "fiscal_year": "GJAHR",
        # Add format-specific mappings dynamically from storage
    }}

    # Apply transformations with type conversion
    for source_key, value in source_data.items():
        target_key = field_map.get(source_key, source_key)

        # Type conversions
        if isinstance(value, str) and target_key in ["WRBTR", "MWSTS"]:
            # Convert amount strings to float
            try:
                target_data[target_key] = float(value.replace(",", ""))
            except (ValueError, AttributeError):
                target_data[target_key] = value
        elif isinstance(value, str) and target_key in ["BLDAT", "BUDAT"]:
            # Convert date strings to SAP format (YYYYMMDD)
            try:
                from datetime import datetime
                dt = datetime.strptime(value, "%Y-%m-%d")
                target_data[target_key] = dt.strftime("%Y%m%d")
            except (ValueError, AttributeError):
                target_data[target_key] = value
        else:
            target_data[target_key] = value

    return target_data
'''
        return code

    def get_similar_documents(
        self, adc_data: Dict[str, Any], k: int = 5
    ) -> List[Dict[str, Any]]:
        """
        Find similar processed documents.

        Args:
            adc_data: ADC data
            k: Number of results

        Returns:
            List of similar documents
        """
        # Build query from ADC data
        query_parts = [
            str(v) for k, v in adc_data.items()
            if k in ["document_type", "vendor_name", "total_amount"]
        ]

        query = " ".join(query_parts)

        # Search APIs (as proxy for similar documents)
        results = self.storage.search_apis(query, k=k)

        return results

    def get_stats(self) -> Dict[str, Any]:
        """
        Get knowledge base statistics.

        Returns:
            Statistics dictionary
        """
        storage_stats = self.storage.get_stats()

        return {
            "storage": storage_stats,
            "total_items": sum(storage_stats.values()),
        }

    def transform_format(
        self, source_data: Dict[str, Any], source_format: str, target_format: str
    ) -> Dict[str, Any]:
        """
        Transform document data from source format to target format.

        Supports transformations between:
        - OCR extracted data → SAP API format
        - SAP API format → Internal storage format
        - Legacy format → Current format

        Uses the FieldMappingManager for database-driven field mappings.
        Falls back to legacy hardcoded mappings if no mapping file found.

        Args:
            source_data: Source document data
            source_format: Source format identifier (e.g., "OCR", "PURCHASE_ORDER", "PurchaseOrder:Standard")
            target_format: Target format identifier (e.g., "SAP_API", "SAP_ODATA")

        Returns:
            Transformed document data in target format
        """
        # Parse source_format to extract document_type and subtype
        # Format can be:
        # - "PurchaseOrder:Standard" (explicit)
        # - "PURCHASE_ORDER" (will use Standard subtype)
        # - "purchase_order_service" (will parse as PurchaseOrder:Service)

        document_type, subtype = self._parse_format_identifier(source_format)

        # Try to use FieldMappingManager first
        mapping = self.field_mapping_manager.get_mapping(
            document_type=document_type,
            subtype=subtype,
            target_format=target_format
        )

        if mapping:
            # Use database-driven mapping
            logger.debug(f"Using mapping: {document_type}:{subtype} → {target_format}")

            # Validate data before transformation (optional)
            is_valid, errors = self.field_mapping_manager.validate_mapping(
                source_data, mapping, strict=False
            )

            if errors:
                logger.warning(f"Validation warnings for {document_type}: {errors}")

            # Transform data
            target_data = self.field_mapping_manager.transform_data(
                source_data, mapping, max_nesting_level=5
            )

            return target_data

        # Fallback to legacy hardcoded mappings
        logger.info(f"No mapping found for {document_type}:{subtype}, using legacy method")

        field_map = self._get_field_mapping(source_format, target_format)

        if not field_map:
            logger.warning(f"No field mapping found for {source_format} → {target_format}")
            return source_data  # Return unchanged if no mapping

        target_data = {}

        # Apply field mappings (legacy method)
        for source_field, target_field_config in field_map.items():
            if source_field not in source_data:
                continue

            source_value = source_data[source_field]

            # Handle both simple string mappings and dict configs
            if isinstance(target_field_config, str):
                # Simple field name mapping
                target_field = target_field_config
                transformations = []
            else:
                # Full config with transformations
                target_field = target_field_config.get("target_field", source_field)
                transformations = target_field_config.get("transformations", [])

            # Apply transformations
            transformed_value = self._apply_transformations(
                source_value,
                transformations
            )

            target_data[target_field] = transformed_value

        # Copy unmapped fields if configured
        copy_unmapped = isinstance(field_map.get("_config"), dict) and field_map["_config"].get("copy_unmapped", False)
        if copy_unmapped:
            for key, value in source_data.items():
                if key not in field_map and key not in target_data:
                    target_data[key] = value

        return target_data

    def _parse_format_identifier(self, format_str: str) -> tuple:
        """
        Parse format identifier to extract document type and subtype.

        Args:
            format_str: Format identifier string

        Returns:
            Tuple of (document_type, subtype)

        Examples:
            "PurchaseOrder:Standard" -> ("PurchaseOrder", "Standard")
            "PURCHASE_ORDER" -> ("PurchaseOrder", "Standard")
            "purchase_order_service" -> ("PurchaseOrder", "Service")
            "supplier_invoice_credit_memo" -> ("SupplierInvoice", "CreditMemo")
        """
        # Handle explicit format: "DocumentType:Subtype"
        if ":" in format_str:
            parts = format_str.split(":", 1)
            return parts[0], parts[1]

        # Parse common format strings
        format_upper = format_str.upper().replace("-", "_")

        # Map common patterns
        type_mapping = {
            "PURCHASE_ORDER": ("PurchaseOrder", "Standard"),
            "PO": ("PurchaseOrder", "Standard"),
            "PURCHASE_ORDER_STANDARD": ("PurchaseOrder", "Standard"),
            "PURCHASE_ORDER_SERVICE": ("PurchaseOrder", "Service"),
            "PURCHASE_ORDER_SUBCONTRACTING": ("PurchaseOrder", "Subcontracting"),
            "PURCHASE_ORDER_CONSIGNMENT": ("PurchaseOrder", "Consignment"),

            "SUPPLIER_INVOICE": ("SupplierInvoice", "Standard"),
            "INVOICE": ("SupplierInvoice", "Standard"),
            "SUPPLIER_INVOICE_STANDARD": ("SupplierInvoice", "Standard"),
            "SUPPLIER_INVOICE_CREDIT_MEMO": ("SupplierInvoice", "CreditMemo"),
            "SUPPLIER_INVOICE_DOWN_PAYMENT": ("SupplierInvoice", "DownPayment"),
            "CREDIT_MEMO": ("SupplierInvoice", "CreditMemo"),

            "GOODS_RECEIPT": ("GoodsReceipt", "PurchaseOrder"),
            "GOODS_RECEIPT_FOR_PO": ("GoodsReceipt", "PurchaseOrder"),
            "GOODS_RECEIPT_RETURN": ("GoodsReceipt", "Return"),
            "GR": ("GoodsReceipt", "PurchaseOrder"),

            "SERVICE_ENTRY_SHEET": ("ServiceEntrySheet", "PurchaseOrder"),
            "SERVICE_ENTRY_SHEET_FOR_PO": ("ServiceEntrySheet", "PurchaseOrder"),
            "SERVICE_ENTRY_SHEET_BLANKET": ("ServiceEntrySheet", "BlanketPO"),
            "SES": ("ServiceEntrySheet", "PurchaseOrder"),

            "PAYMENT_TERMS": ("PaymentTerms", "Standard"),
            "INCOTERMS": ("Incoterms", "Standard"),

            "SALES_ORDER": ("SalesOrder", "Standard"),
            "SO": ("SalesOrder", "Standard"),
        }

        result = type_mapping.get(format_upper)
        if result:
            return result

        # Default: try to convert to PascalCase and use Standard subtype
        # Convert: "purchase_order" -> "PurchaseOrder"
        words = format_str.replace("-", "_").split("_")
        document_type = "".join(word.capitalize() for word in words)

        return document_type, "Standard"

    def _get_field_mapping(
        self, source_format: str, target_format: str
    ) -> Dict[str, Any]:
        """
        Get field mapping configuration for format transformation.

        Mappings loaded from configs or generated dynamically based on
        document type and target system.

        Args:
            source_format: Source format identifier
            target_format: Target format identifier

        Returns:
            Field mapping dictionary
        """
        # Try to get mapping from storage if available
        mapping_key = f"{source_format}_to_{target_format}"

        # Check if we have a stored mapping (future enhancement)
        # For now, generate mappings dynamically

        # Common transformations to SAP API
        if target_format.upper() in ["SAP_API", "SAP_ODATA", "SAP"]:
            return self._get_sap_api_mapping(source_format)

        # Reverse transformation from SAP
        elif source_format.upper() in ["SAP_API", "SAP_ODATA", "SAP"]:
            return self._get_from_sap_mapping(target_format)

        return {}

    def _get_sap_api_mapping(self, source_format: str) -> Dict[str, Any]:
        """
        Get mapping from source format to SAP API format.

        Generates comprehensive field mappings for SAP OData/BAPI calls
        based on document type.

        Args:
            source_format: Source document format/type

        Returns:
            Field mapping dictionary with transformations
        """
        # Base mapping applicable to most SAP documents
        base_mapping = {
            "document_number": {
                "target_field": "DocumentNumber",
                "transformations": ["uppercase", "trim"]
            },
            "document_date": {
                "target_field": "DocumentDate",
                "transformations": ["parse_date", "format_date:YYYYMMDD"]
            },
            "company_code": {
                "target_field": "CompanyCode",
                "transformations": ["pad_left:4:0"]
            },
            "currency": {
                "target_field": "DocumentCurrency",
                "transformations": ["uppercase", "validate_iso_currency"]
            },
        }

        # Document type specific mappings
        source_upper = source_format.upper()

        if "PURCHASE_ORDER" in source_upper or source_upper == "PO":
            base_mapping.update({
                "po_number": {
                    "target_field": "PurchaseOrder",
                    "transformations": ["uppercase", "trim"]
                },
                "purchase_order_number": {
                    "target_field": "PurchaseOrder",
                    "transformations": ["uppercase", "trim"]
                },
                "po_date": {
                    "target_field": "PurchaseOrderDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "purchase_order_date": {
                    "target_field": "PurchaseOrderDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "vendor_id": {
                    "target_field": "Supplier",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "vendor_number": {
                    "target_field": "Supplier",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "supplier": {
                    "target_field": "Supplier",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "requested_delivery_date": {
                    "target_field": "ScheduleLineDeliveryDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "delivery_date": {
                    "target_field": "ScheduleLineDeliveryDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "total_amount": {
                    "target_field": "TotalAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "net_amount": {
                    "target_field": "NetAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "purchasing_organization": {
                    "target_field": "PurchasingOrganization",
                    "transformations": ["pad_left:4:0"]
                },
                "purchasing_group": {
                    "target_field": "PurchasingGroup",
                    "transformations": ["pad_left:3:0"]
                },
                "po_type": {
                    "target_field": "PurchaseOrderType",
                    "transformations": ["uppercase"]
                },
                "payment_terms": {
                    "target_field": "PaymentTerms",
                    "transformations": ["trim"]
                },
            })

        elif "SUPPLIER_INVOICE" in source_upper or "INVOICE" in source_upper:
            base_mapping.update({
                "invoice_number": {
                    "target_field": "SupplierInvoiceIDByInvcgParty",
                    "transformations": ["trim"]
                },
                "supplier_invoice_number": {
                    "target_field": "SupplierInvoiceIDByInvcgParty",
                    "transformations": ["trim"]
                },
                "vendor_invoice_number": {
                    "target_field": "SupplierInvoiceIDByInvcgParty",
                    "transformations": ["trim"]
                },
                "invoice_date": {
                    "target_field": "DocumentDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "posting_date": {
                    "target_field": "PostingDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "vendor_id": {
                    "target_field": "InvoicingParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "vendor_number": {
                    "target_field": "InvoicingParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "supplier": {
                    "target_field": "InvoicingParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "total_amount": {
                    "target_field": "InvoiceGrossAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "gross_amount": {
                    "target_field": "InvoiceGrossAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "tax_amount": {
                    "target_field": "TaxAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "purchase_order": {
                    "target_field": "PurchaseOrder",
                    "transformations": ["uppercase", "trim"]
                },
                "po_number": {
                    "target_field": "PurchaseOrder",
                    "transformations": ["uppercase", "trim"]
                },
                "fiscal_year": {
                    "target_field": "FiscalYear",
                    "transformations": ["trim"]
                },
                "payment_terms": {
                    "target_field": "PaymentTerms",
                    "transformations": ["trim"]
                },
                "payment_method": {
                    "target_field": "PaymentMethod",
                    "transformations": ["uppercase", "trim"]
                },
                "due_date": {
                    "target_field": "DueCalculationBaseDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "baseline_date": {
                    "target_field": "DueCalculationBaseDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
            })

        elif "SALES_ORDER" in source_upper or source_upper == "SO":
            base_mapping.update({
                "sales_order_number": {
                    "target_field": "SalesOrderNumber",
                    "transformations": ["uppercase", "trim"]
                },
                "so_number": {
                    "target_field": "SalesOrderNumber",
                    "transformations": ["uppercase", "trim"]
                },
                "order_date": {
                    "target_field": "SalesOrderDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "sales_order_date": {
                    "target_field": "SalesOrderDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "customer_id": {
                    "target_field": "SoldToParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "customer_number": {
                    "target_field": "SoldToParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "sold_to_party": {
                    "target_field": "SoldToParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "ship_to_party": {
                    "target_field": "ShipToParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "bill_to_party": {
                    "target_field": "BillToParty",
                    "transformations": ["uppercase", "trim", "pad_left:10:0"]
                },
                "customer_po": {
                    "target_field": "PurchaseOrderByCustomer",
                    "transformations": ["trim"]
                },
                "requested_delivery_date": {
                    "target_field": "RequestedDeliveryDate",
                    "transformations": ["parse_date", "format_date:YYYYMMDD"]
                },
                "sales_organization": {
                    "target_field": "SalesOrganization",
                    "transformations": ["pad_left:4:0"]
                },
                "distribution_channel": {
                    "target_field": "DistributionChannel",
                    "transformations": ["pad_left:2:0"]
                },
                "division": {
                    "target_field": "OrganizationDivision",
                    "transformations": ["pad_left:2:0"]
                },
                "total_amount": {
                    "target_field": "TotalAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "net_amount": {
                    "target_field": "NetAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
                "tax_amount": {
                    "target_field": "TaxAmount",
                    "transformations": ["parse_amount", "format_decimal:2"]
                },
            })

        return base_mapping

    def _get_from_sap_mapping(self, target_format: str) -> Dict[str, Any]:
        """
        Get mapping from SAP format to target format.

        Reverse transformation for reading SAP responses.

        Args:
            target_format: Target format identifier

        Returns:
            Field mapping dictionary
        """
        # Reverse common SAP fields to internal format
        reverse_mapping = {
            "DocumentNumber": "document_number",
            "DocumentDate": "document_date",
            "CompanyCode": "company_code",
            "DocumentCurrency": "currency",
            "PurchaseOrder": "purchase_order",
            "Supplier": "vendor_id",
            "InvoicingParty": "vendor_id",
            "SoldToParty": "customer_id",
            "TotalAmount": "total_amount",
            "NetAmount": "net_amount",
            "TaxAmount": "tax_amount",
            "FiscalYear": "fiscal_year",
            "PostingDate": "posting_date",
        }

        return reverse_mapping

    def _apply_transformations(
        self, value: Any, transformations: List[str]
    ) -> Any:
        """
        Apply transformation functions to a value.

        Supported transformations:
        - uppercase, lowercase, trim
        - parse_date, format_date:FORMAT
        - parse_amount, format_decimal:PLACES
        - pad_left:LENGTH:CHAR, pad_right:LENGTH:CHAR
        - validate_iso_currency

        Args:
            value: Value to transform
            transformations: List of transformation strings

        Returns:
            Transformed value
        """
        result = value

        for transformation in transformations:
            try:
                if transformation == "uppercase":
                    result = str(result).upper()

                elif transformation == "lowercase":
                    result = str(result).lower()

                elif transformation == "trim":
                    result = str(result).strip()

                elif transformation == "parse_date":
                    # Parse date from various formats
                    result = self._parse_date_value(result)

                elif transformation.startswith("format_date:"):
                    date_format = transformation.split(":", 1)[1]
                    result = self._format_date_value(result, date_format)

                elif transformation == "parse_amount":
                    # Remove currency symbols, commas
                    result = self._parse_amount_value(result)

                elif transformation.startswith("format_decimal:"):
                    places = int(transformation.split(":", 1)[1])
                    result = round(float(result), places)

                elif transformation.startswith("pad_left:"):
                    parts = transformation.split(":")
                    length = int(parts[1])
                    char = parts[2] if len(parts) > 2 else "0"
                    result = str(result).rjust(length, char)

                elif transformation.startswith("pad_right:"):
                    parts = transformation.split(":")
                    length = int(parts[1])
                    char = parts[2] if len(parts) > 2 else " "
                    result = str(result).ljust(length, char)

                elif transformation == "validate_iso_currency":
                    result = self._validate_currency_value(result)

                else:
                    logger.warning(f"Unknown transformation: {transformation}")

            except Exception as e:
                logger.error(f"Transformation '{transformation}' failed for value '{value}': {e}")
                # Return original value on error
                return value

        return result

    def _parse_date_value(self, date_value: Any) -> datetime:
        """
        Parse date from various formats.

        Args:
            date_value: Date value (string, datetime, or timestamp)

        Returns:
            Parsed datetime object

        Raises:
            ValueError: If date cannot be parsed
        """
        if isinstance(date_value, datetime):
            return date_value

        # Try common date formats
        date_str = str(date_value).strip()

        date_formats = [
            "%Y-%m-%d",           # ISO format: 2024-01-15
            "%d/%m/%Y",           # European: 15/01/2024
            "%m/%d/%Y",           # US: 01/15/2024
            "%Y%m%d",             # SAP format: 20240115
            "%d.%m.%Y",           # German: 15.01.2024
            "%Y-%m-%dT%H:%M:%S",  # ISO with time
            "%Y-%m-%d %H:%M:%S",  # SQL timestamp
        ]

        for fmt in date_formats:
            try:
                return datetime.strptime(date_str, fmt)
            except ValueError:
                continue

        # If all formats fail, raise error
        raise ValueError(f"Cannot parse date: {date_value}")

    def _format_date_value(self, date_value: Any, format_str: str) -> str:
        """
        Format date to specified format.

        Args:
            date_value: Date value (datetime or string)
            format_str: Target format (YYYYMMDD, YYYY-MM-DD, DD/MM/YYYY, or strftime format)

        Returns:
            Formatted date string
        """
        if isinstance(date_value, str):
            date_value = self._parse_date_value(date_value)

        if format_str == "YYYYMMDD":
            return date_value.strftime("%Y%m%d")
        elif format_str == "YYYY-MM-DD":
            return date_value.strftime("%Y-%m-%d")
        elif format_str == "DD/MM/YYYY":
            return date_value.strftime("%d/%m/%Y")
        elif format_str == "MM/DD/YYYY":
            return date_value.strftime("%m/%d/%Y")
        else:
            # Custom format string
            return date_value.strftime(format_str)

    def _parse_amount_value(self, amount_value: Any) -> float:
        """
        Parse amount from various formats.

        Handles currency symbols, thousand separators, etc.

        Args:
            amount_value: Amount value (string, int, or float)

        Returns:
            Parsed float value

        Raises:
            ValueError: If amount cannot be parsed
        """
        import re

        if isinstance(amount_value, (int, float)):
            return float(amount_value)

        # Remove currency symbols and commas
        clean_value = re.sub(r'[^\d.-]', '', str(amount_value))

        try:
            return float(clean_value)
        except (ValueError, AttributeError):
            raise ValueError(f"Cannot parse amount: {amount_value}")

    def _validate_currency_value(self, currency_code: str) -> str:
        """
        Validate and normalize ISO currency code.

        Args:
            currency_code: Currency code to validate

        Returns:
            Validated uppercase currency code

        Raises:
            ValueError: If currency code is invalid
        """
        valid_currencies = [
            "USD", "EUR", "GBP", "JPY", "CHF", "CAD", "AUD", "INR",
            "CNY", "SEK", "NOK", "DKK", "SGD", "HKD", "NZD", "KRW",
            "MXN", "BRL", "ZAR", "RUB", "TRY", "PLN", "THB", "MYR"
        ]

        code = str(currency_code).upper().strip()

        if len(code) != 3:
            raise ValueError(f"Invalid currency code length: {code}")

        if code not in valid_currencies:
            logger.warning(f"Currency code not in common list: {code}")

        return code
