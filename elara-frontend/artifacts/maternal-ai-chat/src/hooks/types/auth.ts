export interface MotherProfile {
  pregnancy_status: string;
  current_week: number;
  expected_due_date: string;
  pregnancies_count: number;
  previous_c_sections: number;
  chronic_conditions: string[];
  blood_type: string;
  allergies: string[];
  key_interests: string[];
}

export interface RegisterPayload {
  email: string;
  password: string;
  full_name: string;
  persona: string;
  language: string;
  mother_profile: MotherProfile;
}
