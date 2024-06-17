package ch07_.inheritance.sec11_sealed;

//The class Docker with a sealed direct superclass or a sealed 
//direct superinterface Person should be declared either final, sealed, or non-sealed
//public class Docker extends Person {

//The type Docker cannot subclass the final class Employee
//public class Docker extends Employee {


//A class Docker declared as non-sealed should have either a sealed direct superclass or 
//a sealed direct superinterface
//public non-sealed class Docker extends Director {

//가능
public class Docker extends Director {
	@Override
	public void work() {
		System.out.println("부두에서 하역을 합니다.");
	}

}
