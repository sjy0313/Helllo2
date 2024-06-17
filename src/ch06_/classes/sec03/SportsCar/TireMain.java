package ch06_.classes.sec03.SportsCar;
//SportsCar 에서 tire class에 public(접근제한자)를 같이 정의하면 아래와 같은 애러 발생 :
// Unresolved compilation problem: The public type Tire must be defined in its own file
// 하나의 파일에서 여러 클래스를 정의할 떄 정의된 클래스의 파일이름이 아닌 경우 public 으로 정의할 수 없다 
//같은 패키지에서 선언된 클래스는 가족이기 떄문에 
//public = 접근제한자 (java에서는 접근의 범위를 엄격하게 관리한다)
public class TireMain {

	public static void main(String[] args) {
		Tire tire = new Tire();
		System.out.println(tire);

	}

}
