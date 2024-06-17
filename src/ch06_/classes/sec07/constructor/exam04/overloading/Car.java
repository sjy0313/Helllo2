package ch06_.classes.sec07.constructor.exam04.overloading;

public class Car {
	//필드 선언
	String company = "현대자동차";
	String model;
	String color;
	int maxSpeed;
	
	// 생성자 선언
		Car() {}
		
		// 생성자를 통해 필드 초기화
		/*
		Car(String company) {
			this.company = company; // -> car1.model에 null 값  
		} */
		
		Car(String model) {
			this.model = model;
		}
		Car(String model, String color) {
			this.model = model;
			this.color = color;
			
		}
		
		Car(String model, String color, String company, int maxSpeed) {
			this.model = model;
			this.color = color;
			this.company = company;
			this.maxSpeed = maxSpeed;
			
		}
	
	Car(String model) { 
		this.model = model; 
	}
	
	Car(String model, String color) {
		this.model = model;
		this.color = color;
	}
	
	Car(String model, String color, int maxSpeed) {
		this.model = model;
		this.color = color;
		this.maxSpeed = maxSpeed;
	}
}